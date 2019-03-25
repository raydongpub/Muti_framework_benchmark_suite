#ifndef _DECL_nbody_H_
#define _DECL_nbody_H_
#include "charm++.h"
#include "envelope.h"
#include <memory>
#include "sdag.h"
/* DECLS: readonly CProxy_Main mainProxy;
 */

/* DECLS: mainchare Main: Chare{
Main(CkArgMsg* impl_msg);
void collect(CkReductionMsg* impl_msg);
void resumeIter(CkReductionMsg* impl_msg);
};
 */
 class Main;
 class CkIndex_Main;
 class CProxy_Main;
/* --------------- index object ------------------ */
class CkIndex_Main:public CkIndex_Chare{
  public:
    typedef Main local_t;
    typedef CkIndex_Main index_t;
    typedef CProxy_Main proxy_t;
    typedef CProxy_Main element_t;

    static int __idx;
    static void __register(const char *s, size_t size);
    /* DECLS: Main(CkArgMsg* impl_msg);
     */
    // Entry point registration at startup
    
    static int reg_Main_CkArgMsg();
    // Entry point index lookup
    
    inline static int idx_Main_CkArgMsg() {
      static int epidx = reg_Main_CkArgMsg();
      return epidx;
    }

    
    static int ckNew(CkArgMsg* impl_msg) { return idx_Main_CkArgMsg(); }
    
    static void _call_Main_CkArgMsg(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_Main_CkArgMsg(void* impl_msg, void* impl_obj);
    /* DECLS: void collect(CkReductionMsg* impl_msg);
     */
    // Entry point registration at startup
    
    static int reg_collect_CkReductionMsg();
    // Entry point index lookup
    
    inline static int idx_collect_CkReductionMsg() {
      static int epidx = reg_collect_CkReductionMsg();
      return epidx;
    }

    
    inline static int idx_collect(void (Main::*)(CkReductionMsg* impl_msg) ) {
      return idx_collect_CkReductionMsg();
    }


    
    static int collect(CkReductionMsg* impl_msg) { return idx_collect_CkReductionMsg(); }
    // Entry point registration at startup
    
    static int reg_redn_wrapper_collect_CkReductionMsg();
    // Entry point index lookup
    
    inline static int idx_redn_wrapper_collect_CkReductionMsg() {
      static int epidx = reg_redn_wrapper_collect_CkReductionMsg();
      return epidx;
    }
    
    static int redn_wrapper_collect(CkReductionMsg* impl_msg) { return idx_redn_wrapper_collect_CkReductionMsg(); }
    
    static void _call_redn_wrapper_collect_CkReductionMsg(void* impl_msg, void* impl_obj_void);
    
    static void _call_collect_CkReductionMsg(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_collect_CkReductionMsg(void* impl_msg, void* impl_obj);
    /* DECLS: void resumeIter(CkReductionMsg* impl_msg);
     */
    // Entry point registration at startup
    
    static int reg_resumeIter_CkReductionMsg();
    // Entry point index lookup
    
    inline static int idx_resumeIter_CkReductionMsg() {
      static int epidx = reg_resumeIter_CkReductionMsg();
      return epidx;
    }

    
    inline static int idx_resumeIter(void (Main::*)(CkReductionMsg* impl_msg) ) {
      return idx_resumeIter_CkReductionMsg();
    }


    
    static int resumeIter(CkReductionMsg* impl_msg) { return idx_resumeIter_CkReductionMsg(); }
    // Entry point registration at startup
    
    static int reg_redn_wrapper_resumeIter_CkReductionMsg();
    // Entry point index lookup
    
    inline static int idx_redn_wrapper_resumeIter_CkReductionMsg() {
      static int epidx = reg_redn_wrapper_resumeIter_CkReductionMsg();
      return epidx;
    }
    
    static int redn_wrapper_resumeIter(CkReductionMsg* impl_msg) { return idx_redn_wrapper_resumeIter_CkReductionMsg(); }
    
    static void _call_redn_wrapper_resumeIter_CkReductionMsg(void* impl_msg, void* impl_obj_void);
    
    static void _call_resumeIter_CkReductionMsg(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_resumeIter_CkReductionMsg(void* impl_msg, void* impl_obj);
};
/* --------------- element proxy ------------------ */
class CProxy_Main:public CProxy_Chare{
  public:
    typedef Main local_t;
    typedef CkIndex_Main index_t;
    typedef CProxy_Main proxy_t;
    typedef CProxy_Main element_t;

    CProxy_Main(void) {};
    CProxy_Main(CkChareID __cid) : CProxy_Chare(__cid){  }
    CProxy_Main(const Chare *c) : CProxy_Chare(c){  }

    int ckIsDelegated(void) const
    { return CProxy_Chare::ckIsDelegated(); }
    inline CkDelegateMgr *ckDelegatedTo(void) const
    { return CProxy_Chare::ckDelegatedTo(); }
    inline CkDelegateData *ckDelegatedPtr(void) const
    { return CProxy_Chare::ckDelegatedPtr(); }
    CkGroupID ckDelegatedIdx(void) const
    { return CProxy_Chare::ckDelegatedIdx(); }

    inline void ckCheck(void) const
    { CProxy_Chare::ckCheck(); }
    const CkChareID &ckGetChareID(void) const
    { return CProxy_Chare::ckGetChareID(); }
    operator const CkChareID &(void) const
    { return ckGetChareID(); }

    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL)
    {       CProxy_Chare::ckDelegate(dTo,dPtr); }
    void ckUndelegate(void)
    {       CProxy_Chare::ckUndelegate(); }
    void pup(PUP::er &p)
    {       CProxy_Chare::pup(p);
    }

    void ckSetChareID(const CkChareID &c)
    {      CProxy_Chare::ckSetChareID(c); }
    Main *ckLocal(void) const
    { return (Main *)CkLocalChare(&ckGetChareID()); }
/* DECLS: Main(CkArgMsg* impl_msg);
 */
    static CkChareID ckNew(CkArgMsg* impl_msg, int onPE=CK_PE_ANY);
    static void ckNew(CkArgMsg* impl_msg, CkChareID* pcid, int onPE=CK_PE_ANY);
    CProxy_Main(CkArgMsg* impl_msg, int onPE=CK_PE_ANY);

/* DECLS: void collect(CkReductionMsg* impl_msg);
 */
    
    void collect(CkReductionMsg* impl_msg);

/* DECLS: void resumeIter(CkReductionMsg* impl_msg);
 */
    
    void resumeIter(CkReductionMsg* impl_msg);

};
PUPmarshall(CProxy_Main)
#define Main_SDAG_CODE 
typedef CBaseT1<Chare, CProxy_Main>CBase_Main;

#include "nbody.h"

/* DECLS: array Data: ArrayElement{
Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
void pauseForLB();
void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
void CudaCompute();
void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
Data(CkMigrateMessage* impl_msg);
};
 */
 class Data;
 class CkIndex_Data;
 class CProxy_Data;
 class CProxyElement_Data;
 class CProxySection_Data;
/* --------------- index object ------------------ */
class CkIndex_Data:public CkIndex_ArrayElement{
  public:
    typedef Data local_t;
    typedef CkIndex_Data index_t;
    typedef CProxy_Data proxy_t;
    typedef CProxyElement_Data element_t;
    typedef CProxySection_Data section_t;

    static int __idx;
    static void __register(const char *s, size_t size);
    /* DECLS: Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
     */
    // Entry point registration at startup
    
    static int reg_Data_marshall1();
    // Entry point index lookup
    
    inline static int idx_Data_marshall1() {
      static int epidx = reg_Data_marshall1();
      return epidx;
    }

    
    static int ckNew(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_) { return idx_Data_marshall1(); }
    
    static void _call_Data_marshall1(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_Data_marshall1(void* impl_msg, void* impl_obj);
    
    static int _callmarshall_Data_marshall1(char* impl_buf, void* impl_obj_void);
    
    static void _marshallmessagepup_Data_marshall1(PUP::er &p,void *msg);
    /* DECLS: void pauseForLB();
     */
    // Entry point registration at startup
    
    static int reg_pauseForLB_void();
    // Entry point index lookup
    
    inline static int idx_pauseForLB_void() {
      static int epidx = reg_pauseForLB_void();
      return epidx;
    }

    
    inline static int idx_pauseForLB(void (Data::*)() ) {
      return idx_pauseForLB_void();
    }


    
    static int pauseForLB() { return idx_pauseForLB_void(); }
    
    static void _call_pauseForLB_void(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_pauseForLB_void(void* impl_msg, void* impl_obj);
    /* DECLS: void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
     */
    // Entry point registration at startup
    
    static int reg_DataLoad_marshall3();
    // Entry point index lookup
    
    inline static int idx_DataLoad_marshall3() {
      static int epidx = reg_DataLoad_marshall3();
      return epidx;
    }

    
    inline static int idx_DataLoad(void (Data::*)(const ParticleDataset::Particle *dPar, int numPar) ) {
      return idx_DataLoad_marshall3();
    }


    
    static int DataLoad(const ParticleDataset::Particle *dPar, int numPar) { return idx_DataLoad_marshall3(); }
    
    static void _call_DataLoad_marshall3(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_DataLoad_marshall3(void* impl_msg, void* impl_obj);
    
    static int _callmarshall_DataLoad_marshall3(char* impl_buf, void* impl_obj_void);
    
    static void _marshallmessagepup_DataLoad_marshall3(PUP::er &p,void *msg);
    /* DECLS: void CudaCompute();
     */
    // Entry point registration at startup
    
    static int reg_CudaCompute_void();
    // Entry point index lookup
    
    inline static int idx_CudaCompute_void() {
      static int epidx = reg_CudaCompute_void();
      return epidx;
    }

    
    inline static int idx_CudaCompute(void (Data::*)() ) {
      return idx_CudaCompute_void();
    }


    
    static int CudaCompute() { return idx_CudaCompute_void(); }
    
    static void _call_CudaCompute_void(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_CudaCompute_void(void* impl_msg, void* impl_obj);
    /* DECLS: void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
     */
    // Entry point registration at startup
    
    static int reg_IterBegin_marshall5();
    // Entry point index lookup
    
    inline static int idx_IterBegin_marshall5() {
      static int epidx = reg_IterBegin_marshall5();
      return epidx;
    }

    
    inline static int idx_IterBegin(void (Data::*)(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect) ) {
      return idx_IterBegin_marshall5();
    }


    
    static int IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect) { return idx_IterBegin_marshall5(); }
    
    static void _call_IterBegin_marshall5(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_IterBegin_marshall5(void* impl_msg, void* impl_obj);
    
    static int _callmarshall_IterBegin_marshall5(char* impl_buf, void* impl_obj_void);
    
    static void _marshallmessagepup_IterBegin_marshall5(PUP::er &p,void *msg);
    /* DECLS: Data(CkMigrateMessage* impl_msg);
     */
    // Entry point registration at startup
    
    static int reg_Data_CkMigrateMessage();
    // Entry point index lookup
    
    inline static int idx_Data_CkMigrateMessage() {
      static int epidx = reg_Data_CkMigrateMessage();
      return epidx;
    }

    
    static int ckNew(CkMigrateMessage* impl_msg) { return idx_Data_CkMigrateMessage(); }
    
    static void _call_Data_CkMigrateMessage(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_Data_CkMigrateMessage(void* impl_msg, void* impl_obj);
};
/* --------------- element proxy ------------------ */
 class CProxyElement_Data : public CProxyElement_ArrayElement{
  public:
    typedef Data local_t;
    typedef CkIndex_Data index_t;
    typedef CProxy_Data proxy_t;
    typedef CProxyElement_Data element_t;
    typedef CProxySection_Data section_t;


    /* TRAM aggregators */

    CProxyElement_Data(void) {
    }
    CProxyElement_Data(const ArrayElement *e) : CProxyElement_ArrayElement(e){
    }

    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL)
    {       CProxyElement_ArrayElement::ckDelegate(dTo,dPtr); }
    void ckUndelegate(void)
    {       CProxyElement_ArrayElement::ckUndelegate(); }
    void pup(PUP::er &p)
    {       CProxyElement_ArrayElement::pup(p);
    }

    int ckIsDelegated(void) const
    { return CProxyElement_ArrayElement::ckIsDelegated(); }
    inline CkDelegateMgr *ckDelegatedTo(void) const
    { return CProxyElement_ArrayElement::ckDelegatedTo(); }
    inline CkDelegateData *ckDelegatedPtr(void) const
    { return CProxyElement_ArrayElement::ckDelegatedPtr(); }
    CkGroupID ckDelegatedIdx(void) const
    { return CProxyElement_ArrayElement::ckDelegatedIdx(); }

    inline void ckCheck(void) const
    { CProxyElement_ArrayElement::ckCheck(); }
    inline operator CkArrayID () const
    { return ckGetArrayID(); }
    inline CkArrayID ckGetArrayID(void) const
    { return CProxyElement_ArrayElement::ckGetArrayID(); }
    inline CkArray *ckLocalBranch(void) const
    { return CProxyElement_ArrayElement::ckLocalBranch(); }
    inline CkLocMgr *ckLocMgr(void) const
    { return CProxyElement_ArrayElement::ckLocMgr(); }

    inline static CkArrayID ckCreateEmptyArray(CkArrayOptions opts = CkArrayOptions())
    { return CProxyElement_ArrayElement::ckCreateEmptyArray(opts); }
    inline static void ckCreateEmptyArrayAsync(CkCallback cb, CkArrayOptions opts = CkArrayOptions())
    { CProxyElement_ArrayElement::ckCreateEmptyArrayAsync(cb, opts); }
    inline static CkArrayID ckCreateArray(CkArrayMessage *m,int ctor,const CkArrayOptions &opts)
    { return CProxyElement_ArrayElement::ckCreateArray(m,ctor,opts); }
    inline void ckInsertIdx(CkArrayMessage *m,int ctor,int onPe,const CkArrayIndex &idx)
    { CProxyElement_ArrayElement::ckInsertIdx(m,ctor,onPe,idx); }
    inline void doneInserting(void)
    { CProxyElement_ArrayElement::doneInserting(); }

    inline void ckBroadcast(CkArrayMessage *m, int ep, int opts=0) const
    { CProxyElement_ArrayElement::ckBroadcast(m,ep,opts); }
    inline void setReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxyElement_ArrayElement::setReductionClient(fn,param); }
    inline void ckSetReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxyElement_ArrayElement::ckSetReductionClient(fn,param); }
    inline void ckSetReductionClient(CkCallback *cb) const
    { CProxyElement_ArrayElement::ckSetReductionClient(cb); }

    inline void ckInsert(CkArrayMessage *m,int ctor,int onPe)
    { CProxyElement_ArrayElement::ckInsert(m,ctor,onPe); }
    inline void ckSend(CkArrayMessage *m, int ep, int opts = 0) const
    { CProxyElement_ArrayElement::ckSend(m,ep,opts); }
    inline void *ckSendSync(CkArrayMessage *m, int ep) const
    { return CProxyElement_ArrayElement::ckSendSync(m,ep); }
    inline const CkArrayIndex &ckGetIndex() const
    { return CProxyElement_ArrayElement::ckGetIndex(); }

    Data *ckLocal(void) const
    { return (Data *)CProxyElement_ArrayElement::ckLocal(); }

    CProxyElement_Data(const CkArrayID &aid,const CkArrayIndex1D &idx,CK_DELCTOR_PARAM)
        :CProxyElement_ArrayElement(aid,idx,CK_DELCTOR_ARGS)
    {
}
    CProxyElement_Data(const CkArrayID &aid,const CkArrayIndex1D &idx)
        :CProxyElement_ArrayElement(aid,idx)
    {
}

    CProxyElement_Data(const CkArrayID &aid,const CkArrayIndex &idx,CK_DELCTOR_PARAM)
        :CProxyElement_ArrayElement(aid,idx,CK_DELCTOR_ARGS)
    {
}
    CProxyElement_Data(const CkArrayID &aid,const CkArrayIndex &idx)
        :CProxyElement_ArrayElement(aid,idx)
    {
}
/* DECLS: Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
 */
    
    void insert(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_, int onPE=-1, const CkEntryOptions *impl_e_opts=NULL);
/* DECLS: void pauseForLB();
 */
    
    void pauseForLB(const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
 */
    
    void DataLoad(const ParticleDataset::Particle *dPar, int numPar, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void CudaCompute();
 */
    
    void CudaCompute(const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
 */
    
    void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: Data(CkMigrateMessage* impl_msg);
 */

};
PUPmarshall(CProxyElement_Data)
/* ---------------- collective proxy -------------- */
 class CProxy_Data : public CProxy_ArrayElement{
  public:
    typedef Data local_t;
    typedef CkIndex_Data index_t;
    typedef CProxy_Data proxy_t;
    typedef CProxyElement_Data element_t;
    typedef CProxySection_Data section_t;

    CProxy_Data(void) {
    }
    CProxy_Data(const ArrayElement *e) : CProxy_ArrayElement(e){
    }

    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL)
    {       CProxy_ArrayElement::ckDelegate(dTo,dPtr); }
    void ckUndelegate(void)
    {       CProxy_ArrayElement::ckUndelegate(); }
    void pup(PUP::er &p)
    {       CProxy_ArrayElement::pup(p);
    }

    int ckIsDelegated(void) const
    { return CProxy_ArrayElement::ckIsDelegated(); }
    inline CkDelegateMgr *ckDelegatedTo(void) const
    { return CProxy_ArrayElement::ckDelegatedTo(); }
    inline CkDelegateData *ckDelegatedPtr(void) const
    { return CProxy_ArrayElement::ckDelegatedPtr(); }
    CkGroupID ckDelegatedIdx(void) const
    { return CProxy_ArrayElement::ckDelegatedIdx(); }

    inline void ckCheck(void) const
    { CProxy_ArrayElement::ckCheck(); }
    inline operator CkArrayID () const
    { return ckGetArrayID(); }
    inline CkArrayID ckGetArrayID(void) const
    { return CProxy_ArrayElement::ckGetArrayID(); }
    inline CkArray *ckLocalBranch(void) const
    { return CProxy_ArrayElement::ckLocalBranch(); }
    inline CkLocMgr *ckLocMgr(void) const
    { return CProxy_ArrayElement::ckLocMgr(); }

    inline static CkArrayID ckCreateEmptyArray(CkArrayOptions opts = CkArrayOptions())
    { return CProxy_ArrayElement::ckCreateEmptyArray(opts); }
    inline static void ckCreateEmptyArrayAsync(CkCallback cb, CkArrayOptions opts = CkArrayOptions())
    { CProxy_ArrayElement::ckCreateEmptyArrayAsync(cb, opts); }
    inline static CkArrayID ckCreateArray(CkArrayMessage *m,int ctor,const CkArrayOptions &opts)
    { return CProxy_ArrayElement::ckCreateArray(m,ctor,opts); }
    inline void ckInsertIdx(CkArrayMessage *m,int ctor,int onPe,const CkArrayIndex &idx)
    { CProxy_ArrayElement::ckInsertIdx(m,ctor,onPe,idx); }
    inline void doneInserting(void)
    { CProxy_ArrayElement::doneInserting(); }

    inline void ckBroadcast(CkArrayMessage *m, int ep, int opts=0) const
    { CProxy_ArrayElement::ckBroadcast(m,ep,opts); }
    inline void setReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxy_ArrayElement::setReductionClient(fn,param); }
    inline void ckSetReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxy_ArrayElement::ckSetReductionClient(fn,param); }
    inline void ckSetReductionClient(CkCallback *cb) const
    { CProxy_ArrayElement::ckSetReductionClient(cb); }

    // Empty array construction
    static CkArrayID ckNew(CkArrayOptions opts = CkArrayOptions()) { return ckCreateEmptyArray(opts); }
    static void      ckNew(CkCallback cb, CkArrayOptions opts = CkArrayOptions()) { ckCreateEmptyArrayAsync(cb, opts); }

    // Generalized array indexing:
    CProxyElement_Data operator [] (const CkArrayIndex1D &idx) const
    { return CProxyElement_Data(ckGetArrayID(), idx, CK_DELCTOR_CALL); }
    CProxyElement_Data operator() (const CkArrayIndex1D &idx) const
    { return CProxyElement_Data(ckGetArrayID(), idx, CK_DELCTOR_CALL); }
    CProxyElement_Data operator [] (int idx) const 
        {return CProxyElement_Data(ckGetArrayID(), CkArrayIndex1D(idx), CK_DELCTOR_CALL);}
    CProxyElement_Data operator () (int idx) const 
        {return CProxyElement_Data(ckGetArrayID(), CkArrayIndex1D(idx), CK_DELCTOR_CALL);}
    CProxy_Data(const CkArrayID &aid,CK_DELCTOR_PARAM) 
        :CProxy_ArrayElement(aid,CK_DELCTOR_ARGS) {}
    CProxy_Data(const CkArrayID &aid) 
        :CProxy_ArrayElement(aid) {}
/* DECLS: Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
 */
    
    static CkArrayID ckNew(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_, const CkArrayOptions &opts = CkArrayOptions(), const CkEntryOptions *impl_e_opts=NULL);
    static void      ckNew(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_, const CkArrayOptions &opts, CkCallback _ck_array_creation_cb, const CkEntryOptions *impl_e_opts=NULL);
    static CkArrayID ckNew(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_, const int s1, const CkEntryOptions *impl_e_opts=NULL);
    static void ckNew(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_, const int s1, CkCallback _ck_array_creation_cb, const CkEntryOptions *impl_e_opts=NULL);

/* DECLS: void pauseForLB();
 */
    
    void pauseForLB(const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
 */
    
    void DataLoad(const ParticleDataset::Particle *dPar, int numPar, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void CudaCompute();
 */
    
    void CudaCompute(const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
 */
    
    void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: Data(CkMigrateMessage* impl_msg);
 */

};
PUPmarshall(CProxy_Data)
/* ---------------- section proxy -------------- */
 class CProxySection_Data : public CProxySection_ArrayElement{
  public:
    typedef Data local_t;
    typedef CkIndex_Data index_t;
    typedef CProxy_Data proxy_t;
    typedef CProxyElement_Data element_t;
    typedef CProxySection_Data section_t;

    CProxySection_Data(void) {
    }

    void ckDelegate(CkDelegateMgr *dTo,CkDelegateData *dPtr=NULL)
    {       CProxySection_ArrayElement::ckDelegate(dTo,dPtr); }
    void ckUndelegate(void)
    {       CProxySection_ArrayElement::ckUndelegate(); }
    void pup(PUP::er &p)
    {       CProxySection_ArrayElement::pup(p);
    }

    int ckIsDelegated(void) const
    { return CProxySection_ArrayElement::ckIsDelegated(); }
    inline CkDelegateMgr *ckDelegatedTo(void) const
    { return CProxySection_ArrayElement::ckDelegatedTo(); }
    inline CkDelegateData *ckDelegatedPtr(void) const
    { return CProxySection_ArrayElement::ckDelegatedPtr(); }
    CkGroupID ckDelegatedIdx(void) const
    { return CProxySection_ArrayElement::ckDelegatedIdx(); }

    inline void ckCheck(void) const
    { CProxySection_ArrayElement::ckCheck(); }
    inline operator CkArrayID () const
    { return ckGetArrayID(); }
    inline CkArrayID ckGetArrayID(void) const
    { return CProxySection_ArrayElement::ckGetArrayID(); }
    inline CkArray *ckLocalBranch(void) const
    { return CProxySection_ArrayElement::ckLocalBranch(); }
    inline CkLocMgr *ckLocMgr(void) const
    { return CProxySection_ArrayElement::ckLocMgr(); }

    inline static CkArrayID ckCreateEmptyArray(CkArrayOptions opts = CkArrayOptions())
    { return CProxySection_ArrayElement::ckCreateEmptyArray(opts); }
    inline static void ckCreateEmptyArrayAsync(CkCallback cb, CkArrayOptions opts = CkArrayOptions())
    { CProxySection_ArrayElement::ckCreateEmptyArrayAsync(cb, opts); }
    inline static CkArrayID ckCreateArray(CkArrayMessage *m,int ctor,const CkArrayOptions &opts)
    { return CProxySection_ArrayElement::ckCreateArray(m,ctor,opts); }
    inline void ckInsertIdx(CkArrayMessage *m,int ctor,int onPe,const CkArrayIndex &idx)
    { CProxySection_ArrayElement::ckInsertIdx(m,ctor,onPe,idx); }
    inline void doneInserting(void)
    { CProxySection_ArrayElement::doneInserting(); }

    inline void ckBroadcast(CkArrayMessage *m, int ep, int opts=0) const
    { CProxySection_ArrayElement::ckBroadcast(m,ep,opts); }
    inline void setReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxySection_ArrayElement::setReductionClient(fn,param); }
    inline void ckSetReductionClient(CkReductionClientFn fn,void *param=NULL) const
    { CProxySection_ArrayElement::ckSetReductionClient(fn,param); }
    inline void ckSetReductionClient(CkCallback *cb) const
    { CProxySection_ArrayElement::ckSetReductionClient(cb); }

    inline void ckSend(CkArrayMessage *m, int ep, int opts = 0)
    { CProxySection_ArrayElement::ckSend(m,ep,opts); }
    inline CkSectionInfo &ckGetSectionInfo()
    { return CProxySection_ArrayElement::ckGetSectionInfo(); }
    inline CkSectionID *ckGetSectionIDs()
    { return CProxySection_ArrayElement::ckGetSectionIDs(); }
    inline CkSectionID &ckGetSectionID()
    { return CProxySection_ArrayElement::ckGetSectionID(); }
    inline CkSectionID &ckGetSectionID(int i)
    { return CProxySection_ArrayElement::ckGetSectionID(i); }
    inline CkArrayID ckGetArrayIDn(int i) const
    { return CProxySection_ArrayElement::ckGetArrayIDn(i); } 
    inline CkArrayIndex *ckGetArrayElements() const
    { return CProxySection_ArrayElement::ckGetArrayElements(); }
    inline CkArrayIndex *ckGetArrayElements(int i) const
    { return CProxySection_ArrayElement::ckGetArrayElements(i); }
    inline int ckGetNumElements() const
    { return CProxySection_ArrayElement::ckGetNumElements(); } 
    inline int ckGetNumElements(int i) const
    { return CProxySection_ArrayElement::ckGetNumElements(i); }    // Generalized array indexing:
    CProxyElement_Data operator [] (const CkArrayIndex1D &idx) const
        {return CProxyElement_Data(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_Data operator() (const CkArrayIndex1D &idx) const
        {return CProxyElement_Data(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_Data operator [] (int idx) const 
        {return CProxyElement_Data(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}
    CProxyElement_Data operator () (int idx) const 
        {return CProxyElement_Data(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}
    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex1D *elems, int nElems, int factor=USE_DEFAULT_BRANCH_FACTOR) {
      return CkSectionID(aid, elems, nElems, factor);
    } 
    static CkSectionID ckNew(const CkArrayID &aid, int l, int u, int s, int factor=USE_DEFAULT_BRANCH_FACTOR) {
      CkVec<CkArrayIndex1D> al;
      for (int i=l; i<=u; i+=s) al.push_back(CkArrayIndex1D(i));
      return CkSectionID(aid, al.getVec(), al.size(), factor);
    } 
    CProxySection_Data(const CkArrayID &aid, CkArrayIndex *elems, int nElems, CK_DELCTOR_PARAM) 
        :CProxySection_ArrayElement(aid,elems,nElems,CK_DELCTOR_ARGS) {}
    CProxySection_Data(const CkArrayID &aid, CkArrayIndex *elems, int nElems, int factor=USE_DEFAULT_BRANCH_FACTOR) 
        :CProxySection_ArrayElement(aid,elems,nElems, factor) { ckAutoDelegate(); }
    CProxySection_Data(const CkSectionID &sid)  
        :CProxySection_ArrayElement(sid) { ckAutoDelegate(); }
    CProxySection_Data(int n, const CkArrayID *aid, CkArrayIndex const * const *elems, const int *nElems, CK_DELCTOR_PARAM) 
        :CProxySection_ArrayElement(n,aid,elems,nElems,CK_DELCTOR_ARGS) {}
    CProxySection_Data(int n, const CkArrayID *aid, CkArrayIndex const * const *elems, const int *nElems) 
        :CProxySection_ArrayElement(n,aid,elems,nElems) { ckAutoDelegate(); }
    CProxySection_Data(int n, const CkArrayID *aid, CkArrayIndex const * const *elems, const int *nElems, int factor) 
        :CProxySection_ArrayElement(n,aid,elems,nElems, factor) { ckAutoDelegate(); }
    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex *elems, int nElems) {
      return CkSectionID(aid, elems, nElems);
    } 
    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex *elems, int nElems, int factor) {
      return CkSectionID(aid, elems, nElems, factor);
    } 
    void ckAutoDelegate(int opts=1) {
      if(ckIsDelegated()) return;
      CProxySection_ArrayElement::ckAutoDelegate(opts);
    } 
    void setReductionClient(CkCallback *cb) {
      CProxySection_ArrayElement::setReductionClient(cb);
    } 
    void resetSection() {
      CProxySection_ArrayElement::resetSection();
    } 
    static void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &sid, int userData=-1, int fragSize=-1);
    static void contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &sid, CkCallback &cb, int userData=-1, int fragSize=-1);
/* DECLS: Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
 */
    

/* DECLS: void pauseForLB();
 */
    
    void pauseForLB(const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
 */
    
    void DataLoad(const ParticleDataset::Particle *dPar, int numPar, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void CudaCompute();
 */
    
    void CudaCompute(const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
 */
    
    void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: Data(CkMigrateMessage* impl_msg);
 */

};
PUPmarshall(CProxySection_Data)
#define Data_SDAG_CODE                                                         \
public:                                                                        \
  void IterBegin(ParticleDataset::Particle * dPar, int numPar, CkCallback collect);\
  void IterBegin(Closure_Data::IterBegin_5_closure* gen0);                     \
private:                                                                       \
  void IterBegin_end(Closure_Data::IterBegin_5_closure* gen0);                 \
  void _slist_0(Closure_Data::IterBegin_5_closure* gen0);                      \
  void _slist_0_end(Closure_Data::IterBegin_5_closure* gen0);                  \
  void _serial_0(Closure_Data::IterBegin_5_closure* gen0);                     \
  void _serial_1(Closure_Data::IterBegin_5_closure* gen0);                     \
public:                                                                        \
public:                                                                        \
  SDAG::dep_ptr __dep;                                                         \
  void _sdag_init();                                                           \
  void __sdag_init();                                                          \
public:                                                                        \
  void _sdag_pup(PUP::er &p);                                                  \
  void __sdag_pup(PUP::er &p) { }                                              \
  static void __sdag_register();                                               \
  static int _sdag_idx_Data_serial_0();                                        \
  static int _sdag_reg_Data_serial_0();                                        \
  static int _sdag_idx_Data_serial_1();                                        \
  static int _sdag_reg_Data_serial_1();                                        \

typedef CBaseT1<ArrayElementT<CkIndex1D>, CProxy_Data>CBase_Data;


/* ---------------- method closures -------------- */
class Closure_Main {
  public:



};


/* ---------------- method closures -------------- */
class Closure_Data {
  public:


    struct pauseForLB_2_closure;


    struct DataLoad_3_closure;


    struct CudaCompute_4_closure;


    struct IterBegin_5_closure;


};

extern void _registernbody(void);
extern "C" void CkRegisterMainModule(void);
#endif
