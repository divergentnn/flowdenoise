// flowdenoise kernels (compiled with fast math, as the reference CuPy RawModule: options=("--use_fast_math",)).
// Verbatim from the Python reference (flowdegrain2 kernels.cu, md5 cb3bce26329e47eabb1f1e088964f278); the only change
// is the warp shuffle, which goes through FLOWDENOISE_SHFL_DOWN (portable.h) so that hipcc compiles it.
#include "portable.h"
// GPU-only patch motion and production temporal fusion. All accumulators FP32.
// Specialised separable shared-memory tiles; 32 contiguous lanes, 8 output rows.
template<int L,int AX> __device__ void filter_tile(const float* a,float* b,int W,int H,const float* k,int zero,int pre) {
 const int SW=32+(AX==0?L-1:0),SH=8+(AX==1?L-1:0);
 __shared__ float s[SW*SH];
 int t=threadIdx.y*32+threadIdx.x,bx=blockIdx.x*32,by=blockIdx.y*8;
 a+=blockIdx.z*W*H;b+=blockIdx.z*W*H;
 for(int j=t;j<SW*SH;j+=256){int x=bx+j%SW-(AX==0?L/2:0),y=by+j/SW-(AX==1?L/2:0);
 s[j]=(zero&&(x<0||x>=W||y<0||y>=H))?0:a[max(0,min(y,H-1))*W+max(0,min(x,W-1))];}
 __syncthreads();int x=bx+threadIdx.x,y=by+threadIdx.y;
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
#define FILTER(L,A) extern "C" __global__ void sep##L##_##A(const float* a,float* b,int W,int H,const float* k,int z,int p){filter_tile<L,A>(a,b,W,H,k,z,p);}
FILTER(3,0) FILTER(3,1) FILTER(5,0) FILTER(5,1) FILTER(7,0) FILTER(7,1)
FILTER(8,0) FILTER(8,1) FILTER(19,0) FILTER(19,1) FILTER(37,0) FILTER(37,1)
__device__ __forceinline__ int cl(int a,int n) { return max(0,min(a,n-1)); }
__device__ __forceinline__ float sat(float a) { return fminf(1.f,fmaxf(0.f,a)); }
__device__ float at(const float* a,int x,int y,int W,int H) { return a[cl(y,H)*W+cl(x,W)]; }
__device__ float bil(const float* a,float x,float y,int W,int H) {
 int ix=floorf(x),iy=floorf(y); float dx=x-ix,dy=y-iy;
 return (1-dy)*((1-dx)*at(a,ix,iy,W,H)+dx*at(a,ix+1,iy,W,H))+
 dy*((1-dx)*at(a,ix,iy+1,W,H)+dx*at(a,ix+1,iy+1,W,H));
}
extern "C" __global__ void sep(const float* a,float* b,int W,int H,int N,const float* k,int L,int axis,int zero) {
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
 int x=i%W,y=(i/W)%H,base=i-y*W-x; float v=0;
 for(int j=0;j<L;j++) {int xx=x+(axis==0?j-L/2:0),yy=y+(axis==1?j-L/2:0);
 if(!zero || (xx>=0&&xx<W&&yy>=0&&yy<H))v+=k[j]*a[base+cl(yy,H)*W+cl(xx,W)];}
 b[i]=v;
}
extern "C" __global__ void down(const float* a,float* b,int W,int H) {
 int w=W/2,h=H/2,i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=w*h)return;
 int p=(i/w*2)*W+(i%w*2);b[i]=(a[p]+a[p+1]+a[p+W]+a[p+W+1])*.25f;
}
__device__ float reduce(float x,float* sm) {
 int t=threadIdx.x;for(int d=16;d;d/=2)x+=FLOWDENOISE_SHFL_DOWN(x,d);
 if(!(t&31))sm[t/32]=x;__syncthreads();float s=sm[0]+sm[1];__syncthreads();return s;
}
// Overlapping 8x8 patches, stride four. Coarse exhaustive search, then inverse
// compositional translation fitting, mean-normalised just as DIS patch search.
extern "C" __global__ void motion(const float* C,const float* N,int W,int H,
 const float* pred,int PW,int PH,float scale,float* out,int BW,int BH,int R,int iterations,float step) {
 int bx=blockIdx.x,by=blockIdx.y,t=threadIdx.x,px=bx*4+(t%8),py=by*4+t/8;
 __shared__ float sm[2];__shared__ float v[8];
 float c=at(C,px,py,W,H),gx=(at(C,px+1,py,W,H)-at(C,px-1,py,W,H))*.5f,
 gy=(at(C,px,py+1,W,H)-at(C,px,py-1,W,H))*.5f;
 float cm=reduce(c,sm)/64,gm=reduce(gx,sm)/64,hm=reduce(gy,sm)/64;
 gx-=gm;gy-=hm;float aa=reduce(gx*gx,sm)+.001f,ab=reduce(gx*gy,sm),bb=reduce(gy*gy,sm)+.001f;
 float u=0,vv=0;
 if(pred){float x=(bx+.5f)/scale-.5f,y=(by+.5f)/scale-.5f;
 u=bil(pred,x,y,PW,PH)*scale;vv=bil(pred+PW*PH,x,y,PW,PH)*scale;}
 float best=1e30f,bu=u,bv=vv;
 // Include zero and spatial predictors. Each patch makes deterministic choices.
 for(int cand=0;cand<(pred?6:1);cand++) {
 float cu=u,cv=vv;
 if(cand==1){cu=0;cv=0;}
 if(cand>=2){int xx=cl(bx+(cand==2?-1:cand==3?1:0),BW), yy=cl(by+(cand==4?-1:cand==5?1:0),BH);
 cu=bil(pred,(xx+.5f)/scale-.5f,(yy+.5f)/scale-.5f,PW,PH)*scale;
 cv=bil(pred+PW*PH,(xx+.5f)/scale-.5f,(yy+.5f)/scale-.5f,PW,PH)*scale;}
 float d=bil(N,px+cu,py+cv,W,H)-c,ds=reduce(d,sm),ss=reduce(d*d,sm)-ds*ds/64;
 if(ss<best){best=ss;bu=cu;bv=cv;}
 }
 u=bu;vv=bv;
 for(int oy=-R;oy<=R;oy++)for(int ox=-R;ox<=R;ox++) {
 float d=bil(N,px+u+ox,py+vv+oy,W,H)-c,ds=reduce(d,sm),ss=reduce(d*d,sm)-ds*ds/64;
 ss*=1+.002f*(abs(ox)+abs(oy));
 if(ss<best){best=ss;bu=u+ox;bv=vv+oy;}
 }
 u=bu;vv=bv;float initu=u,initv=vv;
 for(int it=0;it<iterations;it++) {
 float d=bil(N,px+u,py+vv,W,H)-c;
 float ax=reduce(d*gx,sm),ay=reduce(d*gy,sm),det=aa*bb-ab*ab;
 float du=(bb*ax-ab*ay)/det,dv=(aa*ay-ab*ax)/det;
 // Trust region prevents unconstrained flat patches from flying away.
 u-=step*fminf(1.f,fmaxf(-1.f,du));vv-=step*fminf(1.f,fmaxf(-1.f,dv));
 }
 if(hypotf(u-initu,vv-initv)>8){u=initu;vv=initv;}
 if(t==0){out[by*BW+bx]=u;out[BW*BH+by*BW+bx]=vv;}
}
// GPU row-sweep inverse search: spatial propagation without inter-block races.
// Opposite sweep uses the prior completed pass for vertical candidates.
extern "C" __global__ void raster(const float* C,const float* N,int W,int H,
 const float* pred,int PW,int PH,const float* prev,float* out,int BW,int BH,int reverse,int iterations) {
 int by=blockIdx.x,t=threadIdx.x,d=blockIdx.y;__shared__ float sm[2];
 N+=d*W*H;out+=d*2*BW*BH;
 if(pred)pred+=d*2*PW*PH;if(prev)prev+=d*2*BW*BH;
 float lastu=0,lastv=0;
 for(int col=0;col<BW;col++){
 int bx=reverse?BW-1-col:col,px=bx*4+t%8,py=by*4+t/8;
 float c=at(C,px,py,W,H),gx=0,gy=0;
 for(int j=-1;j<=1;j++){float k=j==0?2.f:1.f;
 gx+=k*(at(C,px+1,py+j,W,H)-at(C,px-1,py+j,W,H));
 gy+=k*(at(C,px+j,py+1,W,H)-at(C,px+j,py-1,W,H));}
 float aa=reduce(gx*gx,sm),bb=reduce(gy*gy,sm),ab=reduce(gx*gy,sm);
 float gm=reduce(gx,sm)/64,hm=reduce(gy,sm)/64;
 float u=0,v=0;
 if(prev){u=prev[by*BW+bx];v=prev[BW*BH+by*BW+bx];}
 else if(pred){float xx=(bx*4+4+.5f)/2-.5f,yy=(by*4+4+.5f)/2-.5f;
 u=2*bil(pred,xx,yy,PW,PH);v=2*bil(pred+PW*PH,xx,yy,PW,PH);}
 float best=1e30f,bu=u,bv=v;
 for(int k=0;k<3;k++){
 float cu=u,cv=v;
 if(k==1){if(col==0)continue;cu=lastu;cv=lastv;}
 if(k==2){if(!prev||by+1>=BH)continue;cu=prev[(by+1)*BW+bx];cv=prev[BW*BH+(by+1)*BW+bx];}
 float d=bil(N,px+cu,py+cv,W,H)-c,ds=reduce(d,sm),ss=reduce(d*d,sm)-ds*ds/64;
 if(ss<best){best=ss;bu=cu;bv=cv;}
 }
 u=bu;v=bv;float det=fmaxf(aa*bb-ab*ab,.001f),prior=1e30f;
 for(int it=0;it<iterations;it++){
 float d=bil(N,px+u,py+v,W,H)-c,ds=reduce(d,sm),ss=reduce(d*d,sm)-ds*ds/64;
 float ax=reduce(d*gx,sm)-ds*gm,ay=reduce(d*gy,sm)-ds*hm;
 u-=(bb*ax-ab*ay)/det;v-=(aa*ay-ab*ax)/det;
 if(ss>=prior)break;prior=ss;
 }
 if(hypotf(u-bu,v-bv)>8){u=bu;v=bv;}
 lastu=u;lastv=v;
 if(t==0){out[by*BW+bx]=u;out[BW*BH+by*BW+bx]=v;}
 }
}
extern "C" __global__ void dense(const float* C,const float* N,const float* mv,float* f,int W,int H,int BW,int BH) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=W*H)return;int x=i%W,y=i/W;
 N+=blockIdx.y*W*H;mv+=blockIdx.y*2*BW*BH;f+=blockIdx.y*2*W*H;
 float sx=0,sy=0,sw=0;
 // DIS-style photometric densification over patches covering this pixel.
 for(int by=min(BH-1,max(0,(y-7+3)/4));by<=min(BH-1,y/4);by++)for(int bx=min(BW-1,max(0,(x-7+3)/4));bx<=min(BW-1,x/4);bx++){
 float u=mv[by*BW+bx],v=mv[BW*BH+by*BW+bx];
 float w=1.f/fmaxf(1.f,fabsf(bil(N,x+u,y+v,W,H)-C[i]));sx+=w*u;sy+=w*v;sw+=w;}
 f[i]=sx/fmaxf(sw,1e-6f);f[W*H+i]=sy/fmaxf(sw,1e-6f);
}
__device__ float lanc(float d) {d=fabsf(d);if(d<1e-6f)return 1; if(d>=3)return 0;
 return sinpif(d)*sinpif(d/3)/(3.289868133696453f*d*d);}
__device__ float warp(const float* a,float x,float y,int W,int H) {
 int ix=floorf(x),iy=floorf(y);float wx[6],wy[6],sx=0,sy=0;
 #pragma unroll
 for(int k=0;k<6;k++){wx[k]=lanc(x-ix-k+2);wy[k]=lanc(y-iy-k+2);sx+=wx[k];sy+=wy[k];}
 float s=0;
 #pragma unroll
 for(int j=0;j<6;j++) {float row=0;
 #pragma unroll
 for(int k=0;k<6;k++)row+=wx[k]*at(a,ix+k-2,iy+j-2,W,H);s+=wy[j]*row;}
 return s/(sx*sy);
}
__device__ float2 warp2(const float* a,float x,float y,int W,int H) {
 int ix=floorf(x),iy=floorf(y);float wx[6],wy[6],sx=0,sy=0;
 #pragma unroll
 for(int k=0;k<6;k++){wx[k]=lanc(x-ix-k+2);wy[k]=lanc(y-iy-k+2);sx+=wx[k];sy+=wy[k];}
 float u=0,v=0;
 #pragma unroll
 for(int j=0;j<6;j++) {float ru=0,rv=0;
 #pragma unroll
 for(int k=0;k<6;k++){int p=cl(iy+j-2,H)*W+cl(ix+k-2,W);ru+=wx[k]*a[p];rv+=wx[k]*a[W*H+p];}
 u+=wy[j]*ru;v+=wy[j]*rv;}
 return make_float2(u/(sx*sy),v/(sx*sy));
}
extern "C" __global__ void warp_y(const float* n,const float* c,const float* f,float* yw,float* dd,
 float* speed,int W,int H,int FW,int FH,float scale) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=W*H)return;int x=i%W,y=i/W;
 float xx=(x+.5f)/scale-.5f,yy=(y+.5f)/scale-.5f;
 float u=bil(f,xx,yy,FW,FH)*scale,v=bil(f+FW*FH,xx,yy,FW,FH)*scale;
 float z=warp(n,x+u,y+v,W,H);yw[i]=z;dd[i]=fabsf(z-c[i]);speed[i]=hypotf(u,v);
}
extern "C" __global__ void accum(const float* yw,const float* low,const float* sad,const float* dd,
 const float* spread,const float* speed,float* a,float* ws,float* sq,float* hs,float* ha,float* weights,
 int W,int H,float th,float dist,float fast,float spike) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=W*H)return;
 float tm=th/(1+speed[i]/dist/fast),s=sad[i]*16;
 float w=s<tm?(tm*tm-s*s)/(tm*tm+s*s):0;
 int x=i%W,y=i/W;float mx=0;
 for(int yy=-1;yy<=1;yy++)for(int xx=-1;xx<=1;xx++)mx=fmaxf(mx,at(dd,x+xx,y+yy,W,H));
 float t=spike*1.25f*spread[i];w*=sat((1.5f*t-mx)/fmaxf(.5f*t,.001f));
 a[i]+=w*yw[i];ws[i]+=w;sq[i]+=w*w;float h=yw[i]-low[i];hs[i]+=w*h;ha[i]+=w*fabsf(h);weights[i]=w;
}
// Defer the five full-size temporal accumulator writes until all weights exist.
extern "C" __global__ void weight_only(const float* yw,const float* low,const float* sad,const float* dd,
 const float* spread,const float* speed,float* wt,float* hp,int W,int H,float th,float dist,float fast,float spike) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=W*H)return;
 float tm=th/(1+speed[i]/dist/fast),s=sad[i]*16;
 float w=s<tm?(tm*tm-s*s)/(tm*tm+s*s):0;
 int x=i%W,y=i/W;float mx=0;
 for(int yy=-1;yy<=1;yy++)for(int xx=-1;xx<=1;xx++)mx=fmaxf(mx,at(dd,x+xx,y+yy,W,H));
 float t=spike*1.25f*spread[i];w*=sat((1.5f*t-mx)/fmaxf(.5f*t,.001f));
 wt[i]=w;hp[i]=yw[i]-low[i];
}
extern "C" __global__ void reduce_frames(const float* y,const float* hc,const float* yw,const float* wt,const float* hp,
 float* acc,float* ws,float* sq,float* hs,float* ha,int N,int D) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
 float a=y[i],w=1,s=1,h=hc[i],ab=fabsf(h);
 for(int d=0;d<D;d++){int j=d*N+i;float ww=wt[j],hh=hp[j];a+=ww*yw[j];w+=ww;s+=ww*ww;h+=ww*hh;ab+=ww*fabsf(hh);}
 acc[i]=a;ws[i]=w;sq[i]=s;hs[i]=h;ha[i]=ab;
}
extern "C" __global__ void chroma(const float* n,const float* f,const float* wt,float* a,float* ws,
 int W,int H,int FW,int FH,float scale) {
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
 
