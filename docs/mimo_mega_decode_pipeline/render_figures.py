"""Render documentation figures: python docs/mimo_mega_decode_pipeline/render_figures.py.
Requires matplotlib and numpy; diagrams are source schematics, not hardware traces.
"""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none', 'svg.hashsalt': 'mimo-mega-decode-pipeline'})
BLUE='#c4e0ff'; ORANGE='#ffd19a'; GREEN='#bcebd5'; GREY='#e3e5e8'; RED='#f6b9bd'; PURPLE='#dfcdf7'
def canvas(title, h=7):
    fig, ax = plt.subplots(figsize=(15,h)); ax.set_xlim(0,15); ax.set_ylim(0,h); ax.axis('off')
    ax.set_title(title, fontsize=17, weight='bold', pad=18)
    return fig,ax
def box(ax,x,y,w,h,label,c=BLUE,size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025',fc=c,ec='#536171',lw=1))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size)
def arrow(ax,a,b,label=None):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#536171','lw':1.5})
    if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=10)
def save(fig,name):
    fig.savefig(ROOT/(name+'.svg'),bbox_inches='tight', metadata={'Date': None})
    # Matplotlib adds trailing blanks in SVG paths; normalize for review diffs.
    svg = ROOT / (name + '.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)

fig,ax=canvas('EP8 persistent kernel: dependencies and cross-CTA overlap (SCHEMATIC)',8)
for x,w,t,c in [(0,2.7,'Dispatch\nFP8 + scales + routing',ORANGE),(3.2,2.8,'GEMM1\nSiLU + FP8 quant',BLUE),(6.5,2.4,'GEMM2',GREEN),(9.4,2.4,'Weighted scatter\nP2P store issue',ORANGE),(12.3,2.6,'Local combine\ntop-k reduction',RED)]: box(ax,x,6.2,w,1,t,c)
for a,b in [(2.7,3.2),(6,6.5),(8.9,9.4),(11.8,12.3)]: arrow(ax,(a,6.7),(b,6.7))
ax.text(3,5.8,'PLAN_NV gate',ha='center',size=10); ax.text(6.25,5.8,'per-m-tile ready',ha='center',size=10);ax.text(12,5.8,'all 8 flags',ha='center',size=10)
for y,label,segments in [
 (4.5,'CTA A',[(2,1.5,'G1',BLUE),(3.5,1.2,'G1',BLUE),(4.7,1.5,'G2',GREEN),(6.2,.8,'scatter',ORANGE),(7,1.2,'G2',GREEN),(8.2,.8,'scatter',ORANGE),(9,2,'drain / peer wait',GREY),(11,2,'combine',RED)]),
 (3.4,'CTA B',[(2,2,'G1',BLUE),(4,2.5,'last claimed G1',BLUE),(6.5,1.5,'G2',GREEN),(8,1.6,'scatter',ORANGE),(9.6,1.4,'drain / peer wait',GREY),(11,2,'combine',RED)]),
 (2.3,'CTA C',[(2,2.4,'G1',BLUE),(4.4,1.2,'G2',GREEN),(5.6,1,'scatter',ORANGE),(6.6,2.2,'G2',GREEN),(8.8,.7,'scatter',ORANGE),(9.5,1.5,'drain / peer wait',GREY),(11,2,'combine',RED)])]:
    ax.text(.1,y+.28,label,weight='bold'); box(ax,1,y,.9,.6,'dispatch /\nwait',ORANGE,7)
    for x,w,t,c in segments: box(ax,x,y,w,.6,t,c,9)
for x,t in [(2,'dispatch complete'),(11,'last scatter drained + all-peer flags')]:
    ax.plot([x,x],[2,5.2],'--',color='#66717b'); ax.text(x,1.65,t,ha='center',fontsize=10)
ax.text(7.5,.8,'G1 / G2 overlap: different CTAs finish claimed work at different times.\nG2 / scatter overlap: different CTAs; each tile computes before it scatters.\nWidths are illustrative. Orange is a store-issue phase, NOT measured XGMI busy time.',ha='center',va='center',size=12)
save(fig,'pipeline')

fig,ax=canvas('CTA roles change over time; identities follow atomic tickets',6)
for x,w,t,c in [(0,3,'1 owner / planner',PURPLE),(3.4,4,'64 dispatch producers\n4 waves / CTA',ORANGE),(7.8,7,'239 initial consumers / waiters',BLUE)]: box(ax,x,4.3,w,.85,t,c)
box(ax,1.5,2.6,12,1,'After dispatch: ALL 304 CTAs may join G1 / G2 work pool\n8 XCD-local queues; 256 threads = 4 wave64 per CTA',GREEN)
for x in [1.5,5.4,11.3]: arrow(ax,(x,4.3),(x,3.6))
box(ax,1.5,.65,4,1,'First 48 completed CTAs\nskip final gather',GREY)
box(ax,6,.65,7.5,1,'Last 256 completed CTAs become reducers\n<= 1024 waves; small M leaves many without items',RED)
arrow(ax,(3.5,2.6),(3.5,1.65));arrow(ax,(9.5,2.6),(9.5,1.65))
save(fig,'cta_roles')

fig,ax=canvas('GEMM1: A ping-pong in LDS; B and operands in register SSA (SCHEMATIC)',8)
box(ax,0,5.7,3,1.3,'GPU global memory\nA1 FP8 + E8M0 scales\nW1 packed FP4 + scales',GREY)
box(ax,4,6.3,4,1,'LDS: A1 ping / pong\n+ full A1-scale tile',GREEN)
box(ax,9,6.3,5.8,1,'Register SSA: current A operands\ncurrent / next packed B + scales',BLUE)
arrow(ax,(3,6.6),(4,6.6),'A staging');arrow(ax,(8,6.8),(9,6.8),'LDS reads')
arrow(ax,(3,5.7),(10,5.1),'B VMEM directly into register SSA; no B LDS')
box(ax,4,3.9,10.8,1.4,'Current steady step: call_pipe (N-major)\nA operands loaded upfront; next B[ni] issued before current N group\nprepare_b / MFMA invoked per M fragment; no explicit decoded-B hoist across M',BLUE)
arrow(ax,(11.5,6.3),(11.5,5.3))
box(ax,0,2,4.2,1.2,'Tail step: call()\nlazy A read, no next B fetch\nno explicit cross-M B hoist',PURPLE)
box(ax,4.8,2,5,1.2,'FP32 C accumulators in regs\nSiLU(gate) * up in regs\npost-SiLU FP32 -> LDS Cshuffle',GREEN)
box(ax,10.4,2,4.4,1.2,'Row reads + 32-feature max\nE8 scale + OCP FP8 pack\nA2 / scales -> global',ORANGE)
arrow(ax,(4.2,2.6),(4.8,2.6));arrow(ax,(9.8,2.6),(10.4,2.6));arrow(ax,(9,3.9),(7.3,3.2))
ax.text(7.5,.65,'K256 x 16 for H4096. A(k+1) fetched into registers during current compute,\nthen written to the alternate LDS slot with a barrier. Cshuffle aliases A storage after its last use.\ncall_pipe_reuse_b / call_pipe_am_final exist, but are NOT selected by current lookahead=1 path.',ha='center',va='center',size=12)
save(fig,'gemm1')

fig,ax=canvas('GEMM2: triple-buffered A in LDS + double-buffered B in registers',8)
for x,t,c in [(0,'Global A2 FP8\nGPU / XCD L2',GREY),(5,'LDS A ring\nslot 0 / 1 / 2',GREEN),(10,'Register SSA\nA(k), B(k), B(k+1)',BLUE)]: box(ax,x,6.2,4.5,1.05,t,c)
arrow(ax,(4.5,6.7),(5,6.7));arrow(ax,(9.5,6.7),(10,6.7))
ax.text(7.5,5.8,'W2 FP4 + B scales: global -> registers directly. A2 scales: global -> register prefetch.',ha='center')
for y,t,c in [(4.6,'1. Prefetch B(k+1), B-scale(k+1), A2-scale(k+1) -> next register state',BLUE),(3.65,'2. Barrier; LDS A(k) -> operand regs; VMEM A(k+2) -> regs -> free LDS slot',GREEN),(2.7,'3. prepare_b(K128) once; reuse across two M16 fragments; native MFMA K32',BLUE),(1.75,'4. Rotate B/scale SSA state and A ring; next K256 iteration',PURPLE)]:box(ax,1,y,13,.65,t,c)
ax.text(7.5,.6,'K256 x 8 for I2048. Early per-K32 scaling; s_setprio / sched_barrier around MFMA.\nEpilogue: FP32 regs -> LDS Cshuffle -> route-weight multiply -> BF16 -> P2P store issue.\nSequential instructions are software-interleaved; actual stalls require a current instruction trace.',ha='center',va='center',size=12)
save(fig,'gemm2')

fig,ax=canvas('LDS temporal aliasing: same compute slab reused across phases',6)
for y,t,w,c in [(4.3,'G1: two A1 slots = 16 KiB',6,GREEN),(3.25,'G2: three A2 slots = 24 KiB',9,BLUE),(2.2,'G2 epilogue: FP32 Cshuffle = 16 or 32 KiB',12,ORANGE)]:
    ax.text(0,y+.3,t,fontsize=11);box(ax,5,y,w*.78,.65,'same base address / different lifetime',c,10)
ax.text(7.5,1.2,'Compute slab = max(A1 ping/pong, A2 ring, Cshuffle), not their sum.\nSeparate retained allocation: A1 scales 4096 B + work word 16 B.\nPool control / alignment: 384 B. Total: 29,072 B (G2 N128) or 37,264 B (G2 N256).',ha='center',size=12)
save(fig,'lds_alias')

rows=list(csv.DictReader((ROOT.parent/'mimo_mega_decode_ep8.csv').open()))
fig,ax=plt.subplots(figsize=(12,5.5)); x=np.arange(len(rows));width=.35
for d,off,label,c in [('baseline',-width/2,'MORI EPv1 + AITER A8W4','#4d82bf'),('integrated',width/2,'Current MegaMoEV2','#df9342')]:
    mid=np.array([float(r[d+'_ms']) for r in rows]);lo=np.array([float(r[d+'_p10_ms']) for r in rows]);hi=np.array([float(r[d+'_p90_ms']) for r in rows])
    ax.bar(x+off,mid,width,label=label,color=c);ax.errorbar(x+off,mid,yerr=[mid-lo,hi-mid],fmt='none',ecolor='#333',capsize=3)
ax.set_xticks(x,[r['m'] for r in rows]);ax.set_xlabel('M / rank (EP8)');ax.set_ylabel('Full MoE latency (ms)');ax.set_title('Current branch: median and P10-P90, 18 samples per implementation / M');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'latency')

# Historical diagnostic: phase intersections, not communication hiding efficiency.
hist=json.loads((ROOT/'historical_overlap.json').read_text())['rows']
fig,ax=plt.subplots(figsize=(11,5))
x=np.arange(len(hist)); w=.35
for key,off,label,c in [('gemm1_gemm2_overlap_us',-w/2,'G1 / G2 phase intersection','#4d82bf'),('gemm2_scatter_overlap_us',w/2,'G2 / scatter ISSUE intersection','#df9342')]:
    mid=np.array([r[key]['median'] for r in hist]);lo=np.array([r[key]['min'] for r in hist]);hi=np.array([r[key]['max'] for r in hist])
    ax.bar(x+off,mid,w,label=label,color=c)
    ax.errorbar(x+off,mid,yerr=[mid-lo,hi-mid],fmt='none',ecolor='#333',capsize=3)
ax.set_xticks(x,[str(r['m']) for r in hist]);ax.set_xlabel('M / rank; 24 traces / M (8 ranks x 3 trials)')
ax.set_ylabel('Phase intersection (us); median and min-max')
ax.set_title('HISTORICAL FlyDSL0.4.2 diagnostic - NOT current branch overlap',weight='bold')
ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'historical_overlap')

fig,ax=canvas('Within one CTA: K-step software scheduling (SCHEMATIC, not a measured trace)',8)
for base,title in [(4.5,'GEMM1: current k and lookahead k+1'),(.7,'GEMM2: current k, B k+1, A k+2')]:
    ax.text(0,base+2.2,title,weight='bold',fontsize=13)
    for dy,lab in [(1.4,'VMEM / prefetch'),(.65,'LDS traffic'),(-.1,'MFMA / regs')]:ax.text(0,base+dy+.2,lab,size=10)
# Load windows show issue-to-use opportunities, not independent engines' measured occupancy.
for y,segs in [
(5.9,[(3,2,'scale(k+1), A(k+1)',ORANGE),(5.2,4,'B(k+1) per-N loads',ORANGE)]),
(5.15,[(3,1.8,'A(k) -> regs',GREEN),(10,3,'A(k+1) -> pong',GREEN)]),
(4.4,[(5,4.5,'B(k) decode / MFMA, accumulate C',BLUE),(10,3,'wait / barrier / rotate',GREY)]),
(2.1,[(3,2.7,'B / scales(k+1)',ORANGE),(6.3,2.5,'A(k+2) fetch',ORANGE)]),
(1.35,[(3,1.8,'barrier, A(k) read',GREEN),(6.3,2.5,'A(k+2) -> free slot',GREEN)]),
(.6,[(5,4.5,'prepared B(k), scale / MFMA',BLUE),(10,3,'rotate A/B/scale state',PURPLE)])]:
    for x,w,t,c in segs:box(ax,x,y,w,.5,t,c,9)
ax.text(7.5,3.4,'Future loads issued before current compute finishes; store/barrier determines when buffers can rotate.\nHorizontal overlap illustrates scheduling opportunities, not measured VMEM/MFMA concurrency.',ha='center',size=11)
save(fig,'k_step_schedule')
