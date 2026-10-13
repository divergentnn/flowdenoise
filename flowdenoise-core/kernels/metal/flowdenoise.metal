// flowdenoise kernels in Metal Shading Language: flowdenoise.cu with the CUDA spellings mapped to MSL. Each kernel
// takes its CUDA argument list as one constant struct (same order, C layout; device pointers inside it); the body
// then reads the same names. The warp reductions use SIMD-group shuffles (32 lanes on Apple GPUs, as a CUDA warp) in
// the same order. Compiled with fast math, as the reference's (--use_fast_math). Not bit-exact with CUDA: another
// compiler and other fast-math functions. Moved verbatim from GRAIL (kernels/metal/flowdegrain2.metal).
#include <metal_stdlib>
using namespace metal;

#define KARGS uint3 blockIdx [[threadgroup_position_in_grid]], uint3 threadIdx [[thread_position_in_threadgroup]], \
              uint3 blockDim [[threads_per_threadgroup]]
#define SYNC() threadgroup_barrier(mem_flags::mem_threadgroup)
typedef device const float* cfp;
typedef device float* fp;

static inline int imax(int a, int b) { return a > b ? a : b; }
static inline int imin(int a, int b) { return a < b ? a : b; }
static inline float hypot2(float a, float b) { return sqrt(a * a + b * b); }

// Specialised separable shared-memory tiles; 32 contiguous lanes, 8 output rows.
struct SepArgs { cfp a; fp b; int W, H; cfp k; int zero, pre; };
template<int L,int AX> kernel void filter_tile(constant SepArgs& P [[buffer(0)]], KARGS) {
 constexpr int SW=32+(AX==0?L-1:0),SH=8+(AX==1?L-1:0);
 threadgroup float s[SW*SH];
 cfp a=P.a; fp b=P.b; const int W=P.W,H=P.H; cfp k=P.k; const int zero=P.zero,pre=P.pre;
 int t=threadIdx.y*32+threadIdx.x,bx=blockIdx.x*32,by=blockIdx.y*8;
 a+=blockIdx.z*W*H;b+=blockIdx.z*W*H;
 for(int j=t;j<SW*SH;j+=256){int x=bx+j%SW-(AX==0?L/2:0),y=by+j/SW-(AX==1?L/2:0);
 s[j]=(zero&&(x<0||x>=W||y<0||y>=H))?0:a[imax(0,imin(y,H-1))*W+imax(0,imin(x,W-1))];}
 SYNC();int x=bx+threadIdx.x,y=by+threadIdx.y;
 if(x>=W||y>=H)return;float v=0;
 if(L%2){
 int center=(threadIdx.y+(AX==1?L/2:0))*SW+threadIdx.x+(AX==0?L/2:0);
 v=k[L/2]*s[center];
 #pragma unroll
 for(int j=1;j<=L/2;j++)v+=k[L/2+j]*(s[center+j*(AX==0?1:SW)]+s[center-j*(AX==0?1:SW)]);
 }else{
 #pragma unroll
 for(int j=0;j<L;j++)v+=k[j]*s[(threadIdx.y+(AX==1?j:0))*SW+threadIdx.x+(AX==0?j:0)];
 }
 // Two successive replicate-padded 121 passes differ from a replicate-padded
 // binomial-5 only at the very first/last sample.
 if(L==5&&pre){int p=AX==0?x:y,extent=AX==0?W:H;
 int center=(threadIdx.y+(AX==1?2:0))*SW+threadIdx.x+(AX==0?2:0);
 if(p==0)v+=(s[center+(AX==0?1:SW)]-s[center])*.0625f;
 if(p==extent-1)v+=(s[center-(AX==0?1:SW)]-s[center])*.0625f;}
 b[y*W+x]=v;
}
#define FILTER(L,A) template [[host_name("sep" #L "_" #A)]] kernel void filter_tile<L,A>(constant SepArgs&, uint3, uint3, uint3);
FILTER(3,0) FILTER(3,1) FILTER(5,0) FILTER(5,1) FILTER(7,0) FILTER(7,1)
FILTER(8,0) FILTER(8,1) FILTER(19,0) FILTER(19,1) FILTER(37,0) FILTER(37,1)

static inline int cl(int a,int n) { return imax(0,imin(a,n-1)); }
static inline float sat(float a) { return fmin(1.f,fmax(0.f,a)); }
static inline float at(cfp a,int x,int y,int W,int H) { return a[cl(y,H)*W+cl(x,W)]; }
static inline float bil(cfp a,float x,float y,int W,int H) {
 int ix=(int)floor(x),iy=(int)floor(y); float dx=x-ix,dy=y-iy;
 return (1-dy)*((1-dx)*at(a,ix,iy,W,H)+dx*at(a,ix+1,iy,W,H))+
 dy*((1-dx)*at(a,ix,iy+1,W,H)+dx*at(a,ix+1,iy+1,W,H));
}
struct SepGArgs { cfp a; fp b; int W, H, N; cfp k; int L, axis, zero; };
kernel void sep(constant SepGArgs& P [[buffer(0)]], KARGS) {
 cfp a=P.a; fp b=P.b; const int W=P.W,H=P.H,N=P.N,L=P.L,axis=P.axis,zero=P.zero; cfp k=P.k;
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
 int x=i%W,y=(i/W)%H,base=i-y*W-x; float v=0;
 for(int j=0;j<L;j++) {int xx=x+(axis==0?j-L/2:0),yy=y+(axis==1?j-L/2:0);
 if(!zero || (xx>=0&&xx<W&&yy>=0&&yy<H))v+=k[j]*a[base+cl(yy,H)*W+cl(xx,W)];}
 b[i]=v;
}
struct DownArgs { cfp a; fp b; int W, H; };
kernel void down(constant DownArgs& P [[buffer(0)]], KARGS) {
 cfp a=P.a; fp b=P.b; const int W=P.W,H=P.H;
 int w=W/2,h=H/2,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h)return;
 int p=(i/w*2)*W+(i%w*2);b[i]=(a[p]+a[p+1]+a[p+W]+a[p+W+1])*.25f;
}
// GPU row-sweep inverse search: spatial propagation without inter-block races.
// Opposite sweep uses the prior completed pass for vertical candidates.
// Metal: one SIMD group per 8x8 patch, each lane holding two pixels (rows r and r + 4), so every patch sum is one
// hardware SIMD reduction without threadgroup barriers (CUDA: 64 threads, shuffle tree per warp, then the two warps;
// another summation order). The neighbours' levels come as pointers (ns[d]) instead of one copied stack.
static inline float R(float a,float b) { return simd_sum(a+b); }
struct RasterArgs { cfp C; cfp ns[6]; int W, H; cfp pred; int PW, PH; cfp prev; fp out; int BW, BH, reverse, iterations; };
kernel void raster(constant RasterArgs& P [[buffer(0)]], KARGS) {
 cfp C=P.C,pred=P.pred,prev=P.prev; fp out=P.out;
 const int W=P.W,H=P.H,PW=P.PW,PH=P.PH,BW=P.BW,BH=P.BH,reverse=P.reverse,iterations=P.iterations;
 int by=blockIdx.x,t=threadIdx.x,d=blockIdx.y;
 cfp N=P.ns[d];out+=d*2*BW*BH;
 if(pred)pred+=d*2*PW*PH;if(prev)prev+=d*2*BW*BH;
 float lastu=0,lastv=0;
 for(int col=0;col<BW;col++){
 int bx=reverse?BW-1-col:col,px=bx*4+t%8,py0=by*4+t/8,py1=py0+4;
 float c0=at(C,px,py0,W,H),c1=at(C,px,py1,W,H),gx0=0,gy0=0,gx1=0,gy1=0;
 for(int j=-1;j<=1;j++){float k=j==0?2.f:1.f;
 gx0+=k*(at(C,px+1,py0+j,W,H)-at(C,px-1,py0+j,W,H));
 gy0+=k*(at(C,px+j,py0+1,W,H)-at(C,px+j,py0-1,W,H));
 gx1+=k*(at(C,px+1,py1+j,W,H)-at(C,px-1,py1+j,W,H));
 gy1+=k*(at(C,px+j,py1+1,W,H)-at(C,px+j,py1-1,W,H));}
 float aa=R(gx0*gx0,gx1*gx1),bb=R(gy0*gy0,gy1*gy1),ab=R(gx0*gy0,gx1*gy1);
 float gm=R(gx0,gx1)/64,hm=R(gy0,gy1)/64;
 float u=0,v=0;
 if(prev){u=prev[by*BW+bx];v=prev[BW*BH+by*BW+bx];}
 else if(pred){float xx=(bx*4+4+.5f)/2-.5f,yy=(by*4+4+.5f)/2-.5f;
 u=2*bil(pred,xx,yy,PW,PH);v=2*bil(pred+PW*PH,xx,yy,PW,PH);}
 float best=1e30f,bu=u,bv=v;
 for(int k=0;k<3;k++){
 float cu=u,cv=v;
 if(k==1){if(col==0)continue;cu=lastu;cv=lastv;}
 if(k==2){if(!prev||by+1>=BH)continue;cu=prev[(by+1)*BW+bx];cv=prev[BW*BH+(by+1)*BW+bx];}
 float d0=bil(N,px+cu,py0+cv,W,H)-c0,d1=bil(N,px+cu,py1+cv,W,H)-c1,ds=R(d0,d1),ss=R(d0*d0,d1*d1)-ds*ds/64;
 if(ss<best){best=ss;bu=cu;bv=cv;}
 }
 u=bu;v=bv;float det=fmax(aa*bb-ab*ab,.001f),prior=1e30f;
 for(int it=0;it<iterations;it++){
 float d0=bil(N,px+u,py0+v,W,H)-c0,d1=bil(N,px+u,py1+v,W,H)-c1,ds=R(d0,d1),ss=R(d0*d0,d1*d1)-ds*ds/64;
 float ax=R(d0*gx0,d1*gx1)-ds*gm,ay=R(d0*gy0,d1*gy1)-ds*hm;
 u-=(bb*ax-ab*ay)/det;v-=(aa*ay-ab*ax)/det;
 if(ss>=prior)break;prior=ss;
 }
 if(hypot2(u-bu,v-bv)>8){u=bu;v=bv;}
 lastu=u;lastv=v;
 if(t==0){out[by*BW+bx]=u;out[BW*BH+by*BW+bx]=v;}
 }
}
struct DenseArgs { cfp C; cfp ns[6]; cfp mv; fp f; int W, H, BW, BH; };
kernel void dense(constant DenseArgs& P [[buffer(0)]], KARGS) {
 cfp C=P.C,N=P.ns[blockIdx.y],mv=P.mv; fp f=P.f; const int W=P.W,H=P.H,BW=P.BW,BH=P.BH;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=W*H)return;int x=i%W,y=i/W;
 mv+=blockIdx.y*2*BW*BH;f+=blockIdx.y*2*W*H;
 float sx=0,sy=0,sw=0;
 // DIS-style photometric densification over patches covering this pixel.
 for(int by=imin(BH-1,imax(0,(y-7+3)/4));by<=imin(BH-1,y/4);by++)for(int bx=imin(BW-1,imax(0,(x-7+3)/4));bx<=imin(BW-1,x/4);bx++){
 float u=mv[by*BW+bx],v=mv[BW*BH+by*BW+bx];
 float w=1.f/fmax(1.f,fabs(bil(N,x+u,y+v,W,H)-C[i]));sx+=w*u;sy+=w*v;sw+=w;}
 f[i]=sx/fmax(sw,1e-6f);f[W*H+i]=sy/fmax(sw,1e-6f);
}
static inline float lanc(float d) {d=fabs(d);if(d<1e-6f)return 1; if(d>=3)return 0;
 return sin(M_PI_F*d)*sin(M_PI_F*d/3)/(3.289868133696453f*d*d);}
static inline float warp(cfp a,float x,float y,int W,int H) {
 int ix=(int)floor(x),iy=(int)floor(y);float wx[6],wy[6],sx=0,sy=0;
 #pragma unroll
 for(int k=0;k<6;k++){wx[k]=lanc(x-ix-k+2);wy[k]=lanc(y-iy-k+2);sx+=wx[k];sy+=wy[k];}
 float s=0;
 #pragma unroll
 for(int j=0;j<6;j++) {float row=0;
 #pragma unroll
 for(int k=0;k<6;k++)row+=wx[k]*at(a,ix+k-2,iy+j-2,W,H);s+=wy[j]*row;}
 return s/(sx*sy);
}
static inline float2 warp2(cfp a,float x,float y,int W,int H) {
 int ix=(int)floor(x),iy=(int)floor(y);float wx[6],wy[6],sx=0,sy=0;
 #pragma unroll
 for(int k=0;k<6;k++){wx[k]=lanc(x-ix-k+2);wy[k]=lanc(y-iy-k+2);sx+=wx[k];sy+=wy[k];}
 float u=0,v=0;
 #pragma unroll
 for(int j=0;j<6;j++) {float ru=0,rv=0;
 #pragma unroll
 for(int k=0;k<6;k++){int p=cl(iy+j-2,H)*W+cl(ix+k-2,W);ru+=wx[k]*a[p];rv+=wx[k]*a[W*H+p];}
 u+=wy[j]*ru;v+=wy[j]*rv;}
 return float2(u/(sx*sy),v/(sx*sy));
}
struct WarpYArgs { cfp n; cfp c; cfp f; fp yw; fp dd; fp speed; int W, H, FW, FH; float scale; };
kernel void warp_y(constant WarpYArgs& P [[buffer(0)]], KARGS) {
 cfp n=P.n,c=P.c,f=P.f; fp yw=P.yw,dd=P.dd,speed=P.speed; const int W=P.W,H=P.H,FW=P.FW,FH=P.FH; const float scale=P.scale;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=W*H)return;int x=i%W,y=i/W;
 float xx=(x+.5f)/scale-.5f,yy=(y+.5f)/scale-.5f;
 float u=bil(f,xx,yy,FW,FH)*scale,v=bil(f+FW*FH,xx,yy,FW,FH)*scale;
 float z=warp(n,x+u,y+v,W,H);yw[i]=z;dd[i]=fabs(z-c[i]);speed[i]=hypot2(u,v);
}
// Defer the five full-size temporal accumulator writes until all weights exist.
struct WeightArgs { cfp yw; cfp low; cfp sad; cfp dd; cfp spread; cfp speed; fp wt; fp hp; int W, H; float th, dist, fast, spike; };
kernel void weight_only(constant WeightArgs& P [[buffer(0)]], KARGS) {
 cfp yw=P.yw,low=P.low,sad=P.sad,dd=P.dd,spread=P.spread,speed=P.speed; fp wt=P.wt,hp=P.hp;
 const int W=P.W,H=P.H; const float th=P.th,dist=P.dist,fast=P.fast,spike=P.spike;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=W*H)return;
 float tm=th/(1+speed[i]/dist/fast),s=sad[i]*16;
 float w=s<tm?(tm*tm-s*s)/(tm*tm+s*s):0;
 int x=i%W,y=i/W;float mx=0;
 for(int yy=-1;yy<=1;yy++)for(int xx=-1;xx<=1;xx++)mx=fmax(mx,at(dd,x+xx,y+yy,W,H));
 float t=spike*1.25f*spread[i];w*=sat((1.5f*t-mx)/fmax(.5f*t,.001f));
 wt[i]=w;hp[i]=yw[i]-low[i];
}
struct ReduceArgs { cfp y; cfp hc; cfp yw; cfp wt; cfp hp; fp acc; fp ws; fp sq; fp hs; fp ha; int N, D; };
kernel void reduce_frames(constant ReduceArgs& P [[buffer(0)]], KARGS) {
 cfp y=P.y,hc=P.hc,yw=P.yw,wt=P.wt,hp=P.hp; fp acc=P.acc,ws=P.ws,sq=P.sq,hs=P.hs,ha=P.ha; const int N=P.N,D=P.D;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
 float a=y[i],w=1,s=1,h=hc[i],ab=fabs(h);
 for(int d=0;d<D;d++){int j=d*N+i;float ww=wt[j],hh=hp[j];a+=ww*yw[j];w+=ww;s+=ww*ww;h+=ww*hh;ab+=ww*fabs(hh);}
 acc[i]=a;ws[i]=w;sq[i]=s;hs[i]=h;ha[i]=ab;
}
struct ChromaArgs { cfp n; cfp f; cfp wt; fp a; fp ws; int W, H, FW, FH; float scale; };
kernel void chroma(constant ChromaArgs& P [[buffer(0)]], KARGS) {
 cfp n=P.n,f=P.f,wt=P.wt; fp a=P.a,ws=P.ws; const int W=P.W,H=P.H,FW=P.FW,FH=P.FH; const float scale=P.scale;
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=W*H)return;int x=i%W,y=i/W;
 // Average full-resolution flow over each 2x2 luma footprint (linear sampling
 // is identical except at the clamped outermost half-flow pixel).
 float u=0,v=0;
 for(int yy=0;yy<2;yy++)for(int xx=0;xx<2;xx++){
 float fx=(2*x+xx+.5f)/scale-.5f,fy=(2*y+yy+.5f)/scale-.5f;
 u+=bil(f,fx,fy,FW,FH)*scale*.125f;v+=bil(f+FW*FH,fx,fy,FW,FH)*scale*.125f;}
 int p=2*y*2*W+2*x;float w=(wt[p]+wt[p+1]+wt[p+2*W]+wt[p+2*W+1])*.25f;
 float2 z=warp2(n,x+u,y+v,W,H);a[i]+=w*z.x;a[W*H+i]+=w*z.y;ws[i]+=w;
}
