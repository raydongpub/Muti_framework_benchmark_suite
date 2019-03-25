#ifndef _DECL_test_H_
#define _DECL_test_H_
#include "charm++.h"
#include "envelope.h"
#include <memory>
#include "sdag.h"
/* DECLS: readonly CProxy_Main mainProxy;
 */

/* DECLS: mainchare Main: Chare{
Main(CkArgMsg* impl_msg);
void done();
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
    /* DECLS: void done();
     */
    // Entry point registration at startup
    
    static int reg_done_void();
    // Entry point index lookup
    
    inline static int idx_done_void() {
      static int epidx = reg_done_void();
      return epidx;
    }

    
    inline static int idx_done(void (Main::*)() ) {
      return idx_done_void();
    }


    
    static int done() { return idx_done_void(); }
    
    static void _call_done_void(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_done_void(void* impl_msg, void* impl_obj);
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

/* DECLS: void done();
 */
    
    void done(const CkEntryOptions *impl_e_opts=NULL);

};
PUPmarshall(CProxy_Main)
#define Main_SDAG_CODE 
typedef CBaseT1<Chare, CProxy_Main>CBase_Main;

/* DECLS: array Grid: ArrayElement{
Grid(const bool &accept, int work_num);
void SendInput(const CProxy_Grid &output);
void printout(int num, const CkCallback &cb);
void printmain(int num);
void input(int c_num, const int *src);
void update_res(CkReductionMsg* impl_msg);
Grid(CkMigrateMessage* impl_msg);
};
 */
 class Grid;
 class CkIndex_Grid;
 class CProxy_Grid;
 class CProxyElement_Grid;
 class CProxySection_Grid;
/* --------------- index object ------------------ */
class CkIndex_Grid:public CkIndex_ArrayElement{
  public:
    typedef Grid local_t;
    typedef CkIndex_Grid index_t;
    typedef CProxy_Grid proxy_t;
    typedef CProxyElement_Grid element_t;
    typedef CProxySection_Grid section_t;

    static int __idx;
    static void __register(const char *s, size_t size);
    /* DECLS: Grid(const bool &accept, int work_num);
     */
    // Entry point registration at startup
    
    static int reg_Grid_marshall1();
    // Entry point index lookup
    
    inline static int idx_Grid_marshall1() {
      static int epidx = reg_Grid_marshall1();
      return epidx;
    }

    
    static int ckNew(const bool &accept, int work_num) { return idx_Grid_marshall1(); }
    
    static void _call_Grid_marshall1(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_Grid_marshall1(void* impl_msg, void* impl_obj);
    
    static int _callmarshall_Grid_marshall1(char* impl_buf, void* impl_obj_void);
    
    static void _marshallmessagepup_Grid_marshall1(PUP::er &p,void *msg);
    /* DECLS: void SendInput(const CProxy_Grid &output);
     */
    // Entry point registration at startup
    
    static int reg_SendInput_marshall2();
    // Entry point index lookup
    
    inline static int idx_SendInput_marshall2() {
      static int epidx = reg_SendInput_marshall2();
      return epidx;
    }

    
    inline static int idx_SendInput(void (Grid::*)(const CProxy_Grid &output) ) {
      return idx_SendInput_marshall2();
    }


    
    static int SendInput(const CProxy_Grid &output) { return idx_SendInput_marshall2(); }
    
    static void _call_SendInput_marshall2(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_SendInput_marshall2(void* impl_msg, void* impl_obj);
    
    static int _callmarshall_SendInput_marshall2(char* impl_buf, void* impl_obj_void);
    
    static void _marshallmessagepup_SendInput_marshall2(PUP::er &p,void *msg);
    /* DECLS: void printout(int num, const CkCallback &cb);
     */
    // Entry point registration at startup
    
    static int reg_printout_marshall3();
    // Entry point index lookup
    
    inline static int idx_printout_marshall3() {
      static int epidx = reg_printout_marshall3();
      return epidx;
    }

    
    inline static int idx_printout(void (Grid::*)(int num, const CkCallback &cb) ) {
      return idx_printout_marshall3();
    }


    
    static int printout(int num, const CkCallback &cb) { return idx_printout_marshall3(); }
    
    static void _call_printout_marshall3(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_printout_marshall3(void* impl_msg, void* impl_obj);
    
    static int _callmarshall_printout_marshall3(char* impl_buf, void* impl_obj_void);
    
    static void _marshallmessagepup_printout_marshall3(PUP::er &p,void *msg);
    /* DECLS: void printmain(int num);
     */
    // Entry point registration at startup
    
    static int reg_printmain_marshall4();
    // Entry point index lookup
    
    inline static int idx_printmain_marshall4() {
      static int epidx = reg_printmain_marshall4();
      return epidx;
    }

    
    inline static int idx_printmain(void (Grid::*)(int num) ) {
      return idx_printmain_marshall4();
    }


    
    static int printmain(int num) { return idx_printmain_marshall4(); }
    
    static void _call_printmain_marshall4(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_printmain_marshall4(void* impl_msg, void* impl_obj);
    
    static int _callmarshall_printmain_marshall4(char* impl_buf, void* impl_obj_void);
    
    static void _marshallmessagepup_printmain_marshall4(PUP::er &p,void *msg);
    /* DECLS: void input(int c_num, const int *src);
     */
    // Entry point registration at startup
    
    static int reg_input_marshall5();
    // Entry point index lookup
    
    inline static int idx_input_marshall5() {
      static int epidx = reg_input_marshall5();
      return epidx;
    }

    
    inline static int idx_input(void (Grid::*)(int c_num, const int *src) ) {
      return idx_input_marshall5();
    }


    
    static int input(int c_num, const int *src) { return idx_input_marshall5(); }
    
    static void _call_input_marshall5(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_input_marshall5(void* impl_msg, void* impl_obj);
    
    static int _callmarshall_input_marshall5(char* impl_buf, void* impl_obj_void);
    
    static void _marshallmessagepup_input_marshall5(PUP::er &p,void *msg);
    /* DECLS: void update_res(CkReductionMsg* impl_msg);
     */
    // Entry point registration at startup
    
    static int reg_update_res_CkReductionMsg();
    // Entry point index lookup
    
    inline static int idx_update_res_CkReductionMsg() {
      static int epidx = reg_update_res_CkReductionMsg();
      return epidx;
    }

    
    inline static int idx_update_res(void (Grid::*)(CkReductionMsg* impl_msg) ) {
      return idx_update_res_CkReductionMsg();
    }


    
    static int update_res(CkReductionMsg* impl_msg) { return idx_update_res_CkReductionMsg(); }
    // Entry point registration at startup
    
    static int reg_redn_wrapper_update_res_CkReductionMsg();
    // Entry point index lookup
    
    inline static int idx_redn_wrapper_update_res_CkReductionMsg() {
      static int epidx = reg_redn_wrapper_update_res_CkReductionMsg();
      return epidx;
    }
    
    static int redn_wrapper_update_res(CkReductionMsg* impl_msg) { return idx_redn_wrapper_update_res_CkReductionMsg(); }
    
    static void _call_redn_wrapper_update_res_CkReductionMsg(void* impl_msg, void* impl_obj_void);
    
    static void _call_update_res_CkReductionMsg(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_update_res_CkReductionMsg(void* impl_msg, void* impl_obj);
    /* DECLS: Grid(CkMigrateMessage* impl_msg);
     */
    // Entry point registration at startup
    
    static int reg_Grid_CkMigrateMessage();
    // Entry point index lookup
    
    inline static int idx_Grid_CkMigrateMessage() {
      static int epidx = reg_Grid_CkMigrateMessage();
      return epidx;
    }

    
    static int ckNew(CkMigrateMessage* impl_msg) { return idx_Grid_CkMigrateMessage(); }
    
    static void _call_Grid_CkMigrateMessage(void* impl_msg, void* impl_obj);
    
    static void _call_sdag_Grid_CkMigrateMessage(void* impl_msg, void* impl_obj);
};
/* --------------- element proxy ------------------ */
 class CProxyElement_Grid : public CProxyElement_ArrayElement{
  public:
    typedef Grid local_t;
    typedef CkIndex_Grid index_t;
    typedef CProxy_Grid proxy_t;
    typedef CProxyElement_Grid element_t;
    typedef CProxySection_Grid section_t;


    /* TRAM aggregators */

    CProxyElement_Grid(void) {
    }
    CProxyElement_Grid(const ArrayElement *e) : CProxyElement_ArrayElement(e){
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

    Grid *ckLocal(void) const
    { return (Grid *)CProxyElement_ArrayElement::ckLocal(); }

    CProxyElement_Grid(const CkArrayID &aid,const CkArrayIndex1D &idx,CK_DELCTOR_PARAM)
        :CProxyElement_ArrayElement(aid,idx,CK_DELCTOR_ARGS)
    {
}
    CProxyElement_Grid(const CkArrayID &aid,const CkArrayIndex1D &idx)
        :CProxyElement_ArrayElement(aid,idx)
    {
}

    CProxyElement_Grid(const CkArrayID &aid,const CkArrayIndex &idx,CK_DELCTOR_PARAM)
        :CProxyElement_ArrayElement(aid,idx,CK_DELCTOR_ARGS)
    {
}
    CProxyElement_Grid(const CkArrayID &aid,const CkArrayIndex &idx)
        :CProxyElement_ArrayElement(aid,idx)
    {
}
/* DECLS: Grid(const bool &accept, int work_num);
 */
    
    void insert(const bool &accept, int work_num, int onPE=-1, const CkEntryOptions *impl_e_opts=NULL);
/* DECLS: void SendInput(const CProxy_Grid &output);
 */
    
    void SendInput(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void printout(int num, const CkCallback &cb);
 */
    
    void printout(int num, const CkCallback &cb, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void printmain(int num);
 */
    
    void printmain(int num, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void input(int c_num, const int *src);
 */
    
    void input(int c_num, const int *src, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void update_res(CkReductionMsg* impl_msg);
 */
    
    void update_res(CkReductionMsg* impl_msg) ;

/* DECLS: Grid(CkMigrateMessage* impl_msg);
 */

};
PUPmarshall(CProxyElement_Grid)
/* ---------------- collective proxy -------------- */
 class CProxy_Grid : public CProxy_ArrayElement{
  public:
    typedef Grid local_t;
    typedef CkIndex_Grid index_t;
    typedef CProxy_Grid proxy_t;
    typedef CProxyElement_Grid element_t;
    typedef CProxySection_Grid section_t;

    CProxy_Grid(void) {
    }
    CProxy_Grid(const ArrayElement *e) : CProxy_ArrayElement(e){
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
    CProxyElement_Grid operator [] (const CkArrayIndex1D &idx) const
    { return CProxyElement_Grid(ckGetArrayID(), idx, CK_DELCTOR_CALL); }
    CProxyElement_Grid operator() (const CkArrayIndex1D &idx) const
    { return CProxyElement_Grid(ckGetArrayID(), idx, CK_DELCTOR_CALL); }
    CProxyElement_Grid operator [] (int idx) const 
        {return CProxyElement_Grid(ckGetArrayID(), CkArrayIndex1D(idx), CK_DELCTOR_CALL);}
    CProxyElement_Grid operator () (int idx) const 
        {return CProxyElement_Grid(ckGetArrayID(), CkArrayIndex1D(idx), CK_DELCTOR_CALL);}
    CProxy_Grid(const CkArrayID &aid,CK_DELCTOR_PARAM) 
        :CProxy_ArrayElement(aid,CK_DELCTOR_ARGS) {}
    CProxy_Grid(const CkArrayID &aid) 
        :CProxy_ArrayElement(aid) {}
/* DECLS: Grid(const bool &accept, int work_num);
 */
    
    static CkArrayID ckNew(const bool &accept, int work_num, const CkArrayOptions &opts = CkArrayOptions(), const CkEntryOptions *impl_e_opts=NULL);
    static void      ckNew(const bool &accept, int work_num, const CkArrayOptions &opts, CkCallback _ck_array_creation_cb, const CkEntryOptions *impl_e_opts=NULL);
    static CkArrayID ckNew(const bool &accept, int work_num, const int s1, const CkEntryOptions *impl_e_opts=NULL);
    static void ckNew(const bool &accept, int work_num, const int s1, CkCallback _ck_array_creation_cb, const CkEntryOptions *impl_e_opts=NULL);

/* DECLS: void SendInput(const CProxy_Grid &output);
 */
    
    void SendInput(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void printout(int num, const CkCallback &cb);
 */
    
    void printout(int num, const CkCallback &cb, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void printmain(int num);
 */
    
    void printmain(int num, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void input(int c_num, const int *src);
 */
    
    void input(int c_num, const int *src, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void update_res(CkReductionMsg* impl_msg);
 */
    
    void update_res(CkReductionMsg* impl_msg) ;

/* DECLS: Grid(CkMigrateMessage* impl_msg);
 */

};
PUPmarshall(CProxy_Grid)
/* ---------------- section proxy -------------- */
 class CProxySection_Grid : public CProxySection_ArrayElement{
  public:
    typedef Grid local_t;
    typedef CkIndex_Grid index_t;
    typedef CProxy_Grid proxy_t;
    typedef CProxyElement_Grid element_t;
    typedef CProxySection_Grid section_t;

    CProxySection_Grid(void) {
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
    CProxyElement_Grid operator [] (const CkArrayIndex1D &idx) const
        {return CProxyElement_Grid(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_Grid operator() (const CkArrayIndex1D &idx) const
        {return CProxyElement_Grid(ckGetArrayID(), idx, CK_DELCTOR_CALL);}
    CProxyElement_Grid operator [] (int idx) const 
        {return CProxyElement_Grid(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}
    CProxyElement_Grid operator () (int idx) const 
        {return CProxyElement_Grid(ckGetArrayID(), *(CkArrayIndex1D*)&ckGetArrayElements()[idx], CK_DELCTOR_CALL);}
    static CkSectionID ckNew(const CkArrayID &aid, CkArrayIndex1D *elems, int nElems, int factor=USE_DEFAULT_BRANCH_FACTOR) {
      return CkSectionID(aid, elems, nElems, factor);
    } 
    static CkSectionID ckNew(const CkArrayID &aid, int l, int u, int s, int factor=USE_DEFAULT_BRANCH_FACTOR) {
      CkVec<CkArrayIndex1D> al;
      for (int i=l; i<=u; i+=s) al.push_back(CkArrayIndex1D(i));
      return CkSectionID(aid, al.getVec(), al.size(), factor);
    } 
    CProxySection_Grid(const CkArrayID &aid, CkArrayIndex *elems, int nElems, CK_DELCTOR_PARAM) 
        :CProxySection_ArrayElement(aid,elems,nElems,CK_DELCTOR_ARGS) {}
    CProxySection_Grid(const CkArrayID &aid, CkArrayIndex *elems, int nElems, int factor=USE_DEFAULT_BRANCH_FACTOR) 
        :CProxySection_ArrayElement(aid,elems,nElems, factor) { ckAutoDelegate(); }
    CProxySection_Grid(const CkSectionID &sid)  
        :CProxySection_ArrayElement(sid) { ckAutoDelegate(); }
    CProxySection_Grid(int n, const CkArrayID *aid, CkArrayIndex const * const *elems, const int *nElems, CK_DELCTOR_PARAM) 
        :CProxySection_ArrayElement(n,aid,elems,nElems,CK_DELCTOR_ARGS) {}
    CProxySection_Grid(int n, const CkArrayID *aid, CkArrayIndex const * const *elems, const int *nElems) 
        :CProxySection_ArrayElement(n,aid,elems,nElems) { ckAutoDelegate(); }
    CProxySection_Grid(int n, const CkArrayID *aid, CkArrayIndex const * const *elems, const int *nElems, int factor) 
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
/* DECLS: Grid(const bool &accept, int work_num);
 */
    

/* DECLS: void SendInput(const CProxy_Grid &output);
 */
    
    void SendInput(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void printout(int num, const CkCallback &cb);
 */
    
    void printout(int num, const CkCallback &cb, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void printmain(int num);
 */
    
    void printmain(int num, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void input(int c_num, const int *src);
 */
    
    void input(int c_num, const int *src, const CkEntryOptions *impl_e_opts=NULL) ;

/* DECLS: void update_res(CkReductionMsg* impl_msg);
 */
    
    void update_res(CkReductionMsg* impl_msg) ;

/* DECLS: Grid(CkMigrateMessage* impl_msg);
 */

};
PUPmarshall(CProxySection_Grid)
#define Grid_SDAG_CODE                                                         \
public:                                                                        \
  void SendInput(CProxy_Grid output);                                          \
  void SendInput(Closure_Grid::SendInput_2_closure* gen0);                     \
private:                                                                       \
  void SendInput_end(Closure_Grid::SendInput_2_closure* gen0);                 \
  void _slist_0(Closure_Grid::SendInput_2_closure* gen0);                      \
  void _slist_0_end(Closure_Grid::SendInput_2_closure* gen0);                  \
  void _atomic_0(Closure_Grid::SendInput_2_closure* gen0);                     \
public:                                                                        \
  void printout(int num, CkCallback & cb);                                     \
  void printout(Closure_Grid::printout_3_closure* gen0);                       \
private:                                                                       \
  void printout_end(Closure_Grid::printout_3_closure* gen0);                   \
  void _slist_1(Closure_Grid::printout_3_closure* gen0);                       \
  void _slist_1_end(Closure_Grid::printout_3_closure* gen0);                   \
  SDAG::Continuation* _when_0(Closure_Grid::printout_3_closure* gen0);         \
  void _when_0_end(Closure_Grid::printout_3_closure* gen0, Closure_Grid::input_5_closure* gen1);\
  void _atomic_1(Closure_Grid::printout_3_closure* gen0, Closure_Grid::input_5_closure* gen1);\
  void _atomic_2(Closure_Grid::printout_3_closure* gen0);                      \
public:                                                                        \
  void printmain(int num);                                                     \
  void printmain(Closure_Grid::printmain_4_closure* gen0);                     \
private:                                                                       \
  void printmain_end(Closure_Grid::printmain_4_closure* gen0);                 \
  void _slist_2(Closure_Grid::printmain_4_closure* gen0);                      \
  void _slist_2_end(Closure_Grid::printmain_4_closure* gen0);                  \
  void _atomic_3(Closure_Grid::printmain_4_closure* gen0);                     \
public:                                                                        \
  void input(Closure_Grid::input_5_closure* genClosure);                       \
  void input(int c_num, int *src);                                             \
public:                                                                        \
  std::auto_ptr<SDAG::Dependency> __dep;                                       \
  void _sdag_init();                                                           \
  void __sdag_init();                                                          \
public:                                                                        \
  void _sdag_pup(PUP::er &p);                                                  \
  void __sdag_pup(PUP::er &p) { }                                              \
  static void __sdag_register();                                               \
  static int _sdag_idx_Grid_atomic_0();                                        \
  static int _sdag_reg_Grid_atomic_0();                                        \
  static int _sdag_idx_Grid_atomic_1();                                        \
  static int _sdag_reg_Grid_atomic_1();                                        \
  static int _sdag_idx_Grid_atomic_2();                                        \
  static int _sdag_reg_Grid_atomic_2();                                        \
  static int _sdag_idx_Grid_atomic_3();                                        \
  static int _sdag_reg_Grid_atomic_3();                                        \

typedef CBaseT1<ArrayElementT<CkIndex1D>, CProxy_Grid>CBase_Grid;


/* ---------------- method closures -------------- */
class Closure_Main {
  public:


    struct done_2_closure;

};

/* ---------------- method closures -------------- */
class Closure_Grid {
  public:


    struct SendInput_2_closure;


    struct printout_3_closure;


    struct printmain_4_closure;


    struct input_5_closure;



};

extern void _registertest(void);
extern "C" void CkRegisterMainModule(void);
#endif
