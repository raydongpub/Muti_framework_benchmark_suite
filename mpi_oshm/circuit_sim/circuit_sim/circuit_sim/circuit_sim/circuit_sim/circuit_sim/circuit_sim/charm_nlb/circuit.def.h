









/* ---------------- method closures -------------- */
#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Main::done_2_closure : public SDAG::Closure {
      

      done_2_closure() {
        init();
      }
      done_2_closure(CkMigrateMessage*) {
        init();
      }
            void pup(PUP::er& __p) {
        packClosure(__p);
      }
      virtual ~done_2_closure() {
      }
      PUPable_decl(SINGLE_ARG(done_2_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Main::post_run_3_closure : public SDAG::Closure {
      

      post_run_3_closure() {
        init();
      }
      post_run_3_closure(CkMigrateMessage*) {
        init();
      }
            void pup(PUP::er& __p) {
        packClosure(__p);
      }
      virtual ~post_run_3_closure() {
      }
      PUPable_decl(SINGLE_ARG(post_run_3_closure));
    };
#endif /* CK_TEMPLATES_ONLY */


/* ---------------- method closures -------------- */
#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::cleanup_2_closure : public SDAG::Closure {
      

      cleanup_2_closure() {
        init();
      }
      cleanup_2_closure(CkMigrateMessage*) {
        init();
      }
            void pup(PUP::er& __p) {
        packClosure(__p);
      }
      virtual ~cleanup_2_closure() {
      }
      PUPable_decl(SINGLE_ARG(cleanup_2_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::SendInput_3_closure : public SDAG::Closure {
      CProxy_Grid output;


      SendInput_3_closure() {
        init();
      }
      SendInput_3_closure(CkMigrateMessage*) {
        init();
      }
      CProxy_Grid & getP0() { return output;}
      void pup(PUP::er& __p) {
        __p | output;
        packClosure(__p);
      }
      virtual ~SendInput_3_closure() {
      }
      PUPable_decl(SINGLE_ARG(SendInput_3_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::SendPost_4_closure : public SDAG::Closure {
      CProxy_Grid output;


      SendPost_4_closure() {
        init();
      }
      SendPost_4_closure(CkMigrateMessage*) {
        init();
      }
      CProxy_Grid & getP0() { return output;}
      void pup(PUP::er& __p) {
        __p | output;
        packClosure(__p);
      }
      virtual ~SendPost_4_closure() {
      }
      PUPable_decl(SINGLE_ARG(SendPost_4_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::SendLoop_5_closure : public SDAG::Closure {
      CProxy_Grid output;


      SendLoop_5_closure() {
        init();
      }
      SendLoop_5_closure(CkMigrateMessage*) {
        init();
      }
      CProxy_Grid & getP0() { return output;}
      void pup(PUP::er& __p) {
        __p | output;
        packClosure(__p);
      }
      virtual ~SendLoop_5_closure() {
      }
      PUPable_decl(SINGLE_ARG(SendLoop_5_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::pgmrun_6_closure : public SDAG::Closure {
      CkCallback cb;


      pgmrun_6_closure() {
        init();
      }
      pgmrun_6_closure(CkMigrateMessage*) {
        init();
      }
      CkCallback & getP0() { return cb;}
      void pup(PUP::er& __p) {
        __p | cb;
        packClosure(__p);
      }
      virtual ~pgmrun_6_closure() {
      }
      PUPable_decl(SINGLE_ARG(pgmrun_6_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::pgmrunloop_7_closure : public SDAG::Closure {
      CkCallback cb;


      pgmrunloop_7_closure() {
        init();
      }
      pgmrunloop_7_closure(CkMigrateMessage*) {
        init();
      }
      CkCallback & getP0() { return cb;}
      void pup(PUP::er& __p) {
        __p | cb;
        packClosure(__p);
      }
      virtual ~pgmrunloop_7_closure() {
      }
      PUPable_decl(SINGLE_ARG(pgmrunloop_7_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::postrun_8_closure : public SDAG::Closure {
      CkCallback cb;


      postrun_8_closure() {
        init();
      }
      postrun_8_closure(CkMigrateMessage*) {
        init();
      }
      CkCallback & getP0() { return cb;}
      void pup(PUP::er& __p) {
        __p | cb;
        packClosure(__p);
      }
      virtual ~postrun_8_closure() {
      }
      PUPable_decl(SINGLE_ARG(postrun_8_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::input_9_closure : public SDAG::Closure {
      int c_num;
      unsigned char *src;

      CkMarshallMsg* _impl_marshall;
      char* _impl_buf_in;
      int _impl_buf_size;

      input_9_closure() {
        init();
        _impl_marshall = 0;
        _impl_buf_in = 0;
        _impl_buf_size = 0;
      }
      input_9_closure(CkMigrateMessage*) {
        init();
        _impl_marshall = 0;
        _impl_buf_in = 0;
        _impl_buf_size = 0;
      }
      int & getP0() { return c_num;}
      unsigned char *& getP1() { return src;}
      void pup(PUP::er& __p) {
        __p | c_num;
        packClosure(__p);
        __p | _impl_buf_size;
        bool hasMsg = (_impl_marshall != 0); __p | hasMsg;
        if (hasMsg) CkPupMessage(__p, (void**)&_impl_marshall);
        else PUParray(__p, _impl_buf_in, _impl_buf_size);
        if (__p.isUnpacking()) {
          char *impl_buf = _impl_marshall ? _impl_marshall->msgBuf : _impl_buf_in;
          PUP::fromMem implP(impl_buf);
  int c_num; implP|c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
          impl_buf+=CK_ALIGN(implP.size(),16);
          src = (unsigned char *)(impl_buf+impl_off_src);
        }
      }
      virtual ~input_9_closure() {
        if (_impl_marshall) CmiFree(UsrToEnv(_impl_marshall));
      }
      PUPable_decl(SINGLE_ARG(input_9_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::input_pos_10_closure : public SDAG::Closure {
      int c_num;
      float *src;

      CkMarshallMsg* _impl_marshall;
      char* _impl_buf_in;
      int _impl_buf_size;

      input_pos_10_closure() {
        init();
        _impl_marshall = 0;
        _impl_buf_in = 0;
        _impl_buf_size = 0;
      }
      input_pos_10_closure(CkMigrateMessage*) {
        init();
        _impl_marshall = 0;
        _impl_buf_in = 0;
        _impl_buf_size = 0;
      }
      int & getP0() { return c_num;}
      float *& getP1() { return src;}
      void pup(PUP::er& __p) {
        __p | c_num;
        packClosure(__p);
        __p | _impl_buf_size;
        bool hasMsg = (_impl_marshall != 0); __p | hasMsg;
        if (hasMsg) CkPupMessage(__p, (void**)&_impl_marshall);
        else PUParray(__p, _impl_buf_in, _impl_buf_size);
        if (__p.isUnpacking()) {
          char *impl_buf = _impl_marshall ? _impl_marshall->msgBuf : _impl_buf_in;
          PUP::fromMem implP(impl_buf);
  int c_num; implP|c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
          impl_buf+=CK_ALIGN(implP.size(),16);
          src = (float *)(impl_buf+impl_off_src);
        }
      }
      virtual ~input_pos_10_closure() {
        if (_impl_marshall) CmiFree(UsrToEnv(_impl_marshall));
      }
      PUPable_decl(SINGLE_ARG(input_pos_10_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */


/* DEFS: readonly CProxy_Main mainProxy;
 */
extern CProxy_Main mainProxy;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_mainProxy(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|mainProxy;
}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: readonly int num_pieces;
 */
extern int num_pieces;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_num_pieces(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|num_pieces;
}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: readonly int max_pe;
 */
extern int max_pe;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_max_pe(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|max_pe;
}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: readonly int number_loops;
 */
extern int number_loops;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_number_loops(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|number_loops;
}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: readonly int nodes_per_piece;
 */
extern int nodes_per_piece;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_nodes_per_piece(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|nodes_per_piece;
}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: readonly int wires_per_piece;
 */
extern int wires_per_piece;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_wires_per_piece(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|wires_per_piece;
}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: readonly int pct_wire_in_piece;
 */
extern int pct_wire_in_piece;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_pct_wire_in_piece(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|pct_wire_in_piece;
}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: readonly int random_seed;
 */
extern int random_seed;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_random_seed(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|random_seed;
}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: readonly int num_blocks;
 */
extern int num_blocks;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_num_blocks(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|num_blocks;
}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: readonly int num_threads;
 */
extern int num_threads;
#ifndef CK_TEMPLATES_ONLY
extern "C" void __xlater_roPup_num_threads(void *_impl_pup_er) {
  PUP::er &_impl_p=*(PUP::er *)_impl_pup_er;
  _impl_p|num_threads;
}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: mainchare Main: Chare{
Main(CkArgMsg* impl_msg);
void done();
void post_run();
};
 */
#ifndef CK_TEMPLATES_ONLY
 int CkIndex_Main::__idx=0;
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
/* DEFS: Main(CkArgMsg* impl_msg);
 */

CkChareID CProxy_Main::ckNew(CkArgMsg* impl_msg, int impl_onPE)
{
  CkChareID impl_ret;
  CkCreateChare(CkIndex_Main::__idx, CkIndex_Main::idx_Main_CkArgMsg(), impl_msg, &impl_ret, impl_onPE);
  return impl_ret;
}

void CProxy_Main::ckNew(CkArgMsg* impl_msg, CkChareID* pcid, int impl_onPE)
{
  CkCreateChare(CkIndex_Main::__idx, CkIndex_Main::idx_Main_CkArgMsg(), impl_msg, pcid, impl_onPE);
}

  CProxy_Main::CProxy_Main(CkArgMsg* impl_msg, int impl_onPE)
{
  CkChareID impl_ret;
  CkCreateChare(CkIndex_Main::__idx, CkIndex_Main::idx_Main_CkArgMsg(), impl_msg, &impl_ret, impl_onPE);
  ckSetChareID(impl_ret);
}

// Entry point registration function

int CkIndex_Main::reg_Main_CkArgMsg() {
  int epidx = CkRegisterEp("Main(CkArgMsg* impl_msg)",
      _call_Main_CkArgMsg, CMessage_CkArgMsg::__idx, __idx, 0);
  CkRegisterMessagePupFn(epidx, (CkMessagePupFn)CkArgMsg::ckDebugPup);
  return epidx;
}


void CkIndex_Main::_call_Main_CkArgMsg(void* impl_msg, void* impl_obj_void)
{
  Main* impl_obj = static_cast<Main *>(impl_obj_void);
  new (impl_obj) Main((CkArgMsg*)impl_msg);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void done();
 */

void CProxy_Main::done(const CkEntryOptions *impl_e_opts)
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  if (ckIsDelegated()) {
    int destPE=CkChareMsgPrep(CkIndex_Main::idx_done_void(), impl_msg, &ckGetChareID());
    if (destPE!=-1) ckDelegatedTo()->ChareSend(ckDelegatedPtr(),CkIndex_Main::idx_done_void(), impl_msg, &ckGetChareID(),destPE);
  }
  else CkSendMsg(CkIndex_Main::idx_done_void(), impl_msg, &ckGetChareID(),0);
}

// Entry point registration function

int CkIndex_Main::reg_done_void() {
  int epidx = CkRegisterEp("done()",
      _call_done_void, 0, __idx, 0);
  return epidx;
}


void CkIndex_Main::_call_done_void(void* impl_msg, void* impl_obj_void)
{
  Main* impl_obj = static_cast<Main *>(impl_obj_void);
  CkFreeSysMsg(impl_msg);
  impl_obj->done();
}
PUPable_def(SINGLE_ARG(Closure_Main::done_2_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void post_run();
 */

void CProxy_Main::post_run(const CkEntryOptions *impl_e_opts)
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  if (ckIsDelegated()) {
    int destPE=CkChareMsgPrep(CkIndex_Main::idx_post_run_void(), impl_msg, &ckGetChareID());
    if (destPE!=-1) ckDelegatedTo()->ChareSend(ckDelegatedPtr(),CkIndex_Main::idx_post_run_void(), impl_msg, &ckGetChareID(),destPE);
  }
  else CkSendMsg(CkIndex_Main::idx_post_run_void(), impl_msg, &ckGetChareID(),0);
}

// Entry point registration function

int CkIndex_Main::reg_post_run_void() {
  int epidx = CkRegisterEp("post_run()",
      _call_post_run_void, 0, __idx, 0);
  return epidx;
}


void CkIndex_Main::_call_post_run_void(void* impl_msg, void* impl_obj_void)
{
  Main* impl_obj = static_cast<Main *>(impl_obj_void);
  CkFreeSysMsg(impl_msg);
  impl_obj->post_run();
}
PUPable_def(SINGLE_ARG(Closure_Main::post_run_3_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void CkIndex_Main::__register(const char *s, size_t size) {
  __idx = CkRegisterChare(s, size, TypeMainChare);
  CkRegisterBase(__idx, CkIndex_Chare::__idx);
  // REG: Main(CkArgMsg* impl_msg);
  idx_Main_CkArgMsg();
  CkRegisterMainChare(__idx, idx_Main_CkArgMsg());

  // REG: void done();
  idx_done_void();

  // REG: void post_run();
  idx_post_run_void();

}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: array Grid: ArrayElement{
Grid(const bool &accept, int num_pieces_);
void cleanup();
void SendInput(const CProxy_Grid &output);
void SendPost(const CProxy_Grid &output);
void SendLoop(const CProxy_Grid &output);
void pgmrun(const CkCallback &cb);
void pgmrunloop(const CkCallback &cb);
void postrun(const CkCallback &cb);
void input(int c_num, const unsigned char *src);
void input_pos(int c_num, const float *src);
void update_post(CkReductionMsg* impl_msg);
void update_res(CkReductionMsg* impl_msg);
Grid(CkMigrateMessage* impl_msg);
};
 */
#ifndef CK_TEMPLATES_ONLY
 int CkIndex_Grid::__idx=0;
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void CProxySection_Grid::contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &sid, int userData, int fragSize)
{
   CkArray *ckarr = CProxy_CkArray(sid.get_aid()).ckLocalBranch();
   CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(ckarr->getmCastMgr()).ckLocalBranch();
   mCastGrp->contribute(dataSize, data, type, sid, userData, fragSize);
}

void CProxySection_Grid::contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &sid, CkCallback &cb, int userData, int fragSize)
{
   CkArray *ckarr = CProxy_CkArray(sid.get_aid()).ckLocalBranch();
   CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(ckarr->getmCastMgr()).ckLocalBranch();
   mCastGrp->contribute(dataSize, data, type, sid, cb, userData, fragSize);
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
/* DEFS: Grid(const bool &accept, int num_pieces_);
 */

void CProxyElement_Grid::insert(const bool &accept, int num_pieces_, int onPE, const CkEntryOptions *impl_e_opts)
{ 
  //Marshall: const bool &accept, int num_pieces_
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|num_pieces_;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|num_pieces_;
  }
   UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
   ckInsert((CkArrayMessage *)impl_msg,CkIndex_Grid::idx_Grid_marshall1(),onPE);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void cleanup();
 */

void CProxyElement_Grid::cleanup(const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_cleanup_void(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void SendInput(const CProxy_Grid &output);
 */

void CProxyElement_Grid::SendInput(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CProxy_Grid &output
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_SendInput_marshall3(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void SendPost(const CProxy_Grid &output);
 */

void CProxyElement_Grid::SendPost(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CProxy_Grid &output
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_SendPost_marshall4(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void SendLoop(const CProxy_Grid &output);
 */

void CProxyElement_Grid::SendLoop(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CProxy_Grid &output
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_SendLoop_marshall5(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void pgmrun(const CkCallback &cb);
 */

void CProxyElement_Grid::pgmrun(const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_pgmrun_marshall6(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void pgmrunloop(const CkCallback &cb);
 */

void CProxyElement_Grid::pgmrunloop(const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_pgmrunloop_marshall7(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void postrun(const CkCallback &cb);
 */

void CProxyElement_Grid::postrun(const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_postrun_marshall8(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void input(int c_num, const unsigned char *src);
 */

void CProxyElement_Grid::input(int c_num, const unsigned char *src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int c_num, const unsigned char *src
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_src, impl_cnt_src;
  impl_off_src=impl_off=CK_ALIGN(impl_off,sizeof(unsigned char));
  impl_off+=(impl_cnt_src=sizeof(unsigned char)*(c_num));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_src,src,impl_cnt_src);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_input_marshall9(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void input_pos(int c_num, const float *src);
 */

void CProxyElement_Grid::input_pos(int c_num, const float *src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int c_num, const float *src
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_src, impl_cnt_src;
  impl_off_src=impl_off=CK_ALIGN(impl_off,sizeof(float));
  impl_off+=(impl_cnt_src=sizeof(float)*(c_num));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_src,src,impl_cnt_src);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_input_pos_marshall10(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void update_post(CkReductionMsg* impl_msg);
 */

void CProxyElement_Grid::update_post(CkReductionMsg* impl_msg) 
{
  ckCheck();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_update_post_CkReductionMsg(),0);
}

void CkIndex_Grid::_call_redn_wrapper_update_post_CkReductionMsg(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid*> (impl_obj_void);
  char* impl_buf = (char*)((CkReductionMsg*)impl_msg)->getData();
  impl_obj->update_post((CkReductionMsg*)impl_msg);
  delete (CkReductionMsg*)impl_msg;
}

#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void update_res(CkReductionMsg* impl_msg);
 */

void CProxyElement_Grid::update_res(CkReductionMsg* impl_msg) 
{
  ckCheck();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_update_res_CkReductionMsg(),0);
}

void CkIndex_Grid::_call_redn_wrapper_update_res_CkReductionMsg(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid*> (impl_obj_void);
  char* impl_buf = (char*)((CkReductionMsg*)impl_msg)->getData();
  impl_obj->update_res((CkReductionMsg*)impl_msg);
  delete (CkReductionMsg*)impl_msg;
}

#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: Grid(CkMigrateMessage* impl_msg);
 */
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: Grid(const bool &accept, int num_pieces_);
 */

CkArrayID CProxy_Grid::ckNew(const bool &accept, int num_pieces_, const CkArrayOptions &opts, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const bool &accept, int num_pieces_
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|num_pieces_;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|num_pieces_;
  }
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkArrayID gId = ckCreateArray((CkArrayMessage *)impl_msg, CkIndex_Grid::idx_Grid_marshall1(), opts);
  return gId;
}

void CProxy_Grid::ckNew(const bool &accept, int num_pieces_, const CkArrayOptions &opts, CkCallback _ck_array_creation_cb, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const bool &accept, int num_pieces_
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|num_pieces_;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|num_pieces_;
  }
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkSendAsyncCreateArray(CkIndex_Grid::idx_Grid_marshall1(), _ck_array_creation_cb, opts, impl_msg);
}

CkArrayID CProxy_Grid::ckNew(const bool &accept, int num_pieces_, const int s1, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const bool &accept, int num_pieces_
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|num_pieces_;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|num_pieces_;
  }
  CkArrayOptions opts(s1);
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkArrayID gId = ckCreateArray((CkArrayMessage *)impl_msg, CkIndex_Grid::idx_Grid_marshall1(), opts);
  return gId;
}

void CProxy_Grid::ckNew(const bool &accept, int num_pieces_, const int s1, CkCallback _ck_array_creation_cb, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const bool &accept, int num_pieces_
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|num_pieces_;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|num_pieces_;
  }
  CkArrayOptions opts(s1);
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkSendAsyncCreateArray(CkIndex_Grid::idx_Grid_marshall1(), _ck_array_creation_cb, opts, impl_msg);
}

// Entry point registration function

int CkIndex_Grid::reg_Grid_marshall1() {
  int epidx = CkRegisterEp("Grid(const bool &accept, int num_pieces_)",
      _call_Grid_marshall1, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_Grid_marshall1);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_Grid_marshall1);

  return epidx;
}


void CkIndex_Grid::_call_Grid_marshall1(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const bool &accept, int num_pieces_*/
  PUP::fromMem implP(impl_buf);
  bool accept; implP|accept;
  int num_pieces_; implP|num_pieces_;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  new (impl_obj) Grid(accept, num_pieces_);
}

int CkIndex_Grid::_callmarshall_Grid_marshall1(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  /*Unmarshall pup'd fields: const bool &accept, int num_pieces_*/
  PUP::fromMem implP(impl_buf);
  bool accept; implP|accept;
  int num_pieces_; implP|num_pieces_;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  new (impl_obj) Grid(accept, num_pieces_);
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_Grid_marshall1(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const bool &accept, int num_pieces_*/
  PUP::fromMem implP(impl_buf);
  bool accept; implP|accept;
  int num_pieces_; implP|num_pieces_;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("accept");
  implDestP|accept;
  if (implDestP.hasComments()) implDestP.comment("num_pieces_");
  implDestP|num_pieces_;
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void cleanup();
 */

void CProxy_Grid::cleanup(const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_cleanup_void(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_cleanup_void() {
  int epidx = CkRegisterEp("cleanup()",
      _call_cleanup_void, 0, __idx, 0);
  return epidx;
}


void CkIndex_Grid::_call_cleanup_void(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkFreeSysMsg(impl_msg);
  impl_obj->cleanup();
}
PUPable_def(SINGLE_ARG(Closure_Grid::cleanup_2_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void SendInput(const CProxy_Grid &output);
 */

void CProxy_Grid::SendInput(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CProxy_Grid &output
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_SendInput_marshall3(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_SendInput_marshall3() {
  int epidx = CkRegisterEp("SendInput(const CProxy_Grid &output)",
      _call_SendInput_marshall3, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_SendInput_marshall3);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_SendInput_marshall3);

  return epidx;
}


void CkIndex_Grid::_call_SendInput_marshall3(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::SendInput_3_closure* genClosure = new Closure_Grid::SendInput_3_closure();
  implP|genClosure->output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->SendInput(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_SendInput_marshall3(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::SendInput_3_closure* genClosure = new Closure_Grid::SendInput_3_closure();
  implP|genClosure->output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->SendInput(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_SendInput_marshall3(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const CProxy_Grid &output*/
  PUP::fromMem implP(impl_buf);
  CProxy_Grid output; implP|output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("output");
  implDestP|output;
}
PUPable_def(SINGLE_ARG(Closure_Grid::SendInput_3_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void SendPost(const CProxy_Grid &output);
 */

void CProxy_Grid::SendPost(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CProxy_Grid &output
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_SendPost_marshall4(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_SendPost_marshall4() {
  int epidx = CkRegisterEp("SendPost(const CProxy_Grid &output)",
      _call_SendPost_marshall4, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_SendPost_marshall4);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_SendPost_marshall4);

  return epidx;
}


void CkIndex_Grid::_call_SendPost_marshall4(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::SendPost_4_closure* genClosure = new Closure_Grid::SendPost_4_closure();
  implP|genClosure->output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->SendPost(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_SendPost_marshall4(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::SendPost_4_closure* genClosure = new Closure_Grid::SendPost_4_closure();
  implP|genClosure->output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->SendPost(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_SendPost_marshall4(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const CProxy_Grid &output*/
  PUP::fromMem implP(impl_buf);
  CProxy_Grid output; implP|output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("output");
  implDestP|output;
}
PUPable_def(SINGLE_ARG(Closure_Grid::SendPost_4_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void SendLoop(const CProxy_Grid &output);
 */

void CProxy_Grid::SendLoop(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CProxy_Grid &output
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_SendLoop_marshall5(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_SendLoop_marshall5() {
  int epidx = CkRegisterEp("SendLoop(const CProxy_Grid &output)",
      _call_SendLoop_marshall5, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_SendLoop_marshall5);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_SendLoop_marshall5);

  return epidx;
}


void CkIndex_Grid::_call_SendLoop_marshall5(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::SendLoop_5_closure* genClosure = new Closure_Grid::SendLoop_5_closure();
  implP|genClosure->output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->SendLoop(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_SendLoop_marshall5(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::SendLoop_5_closure* genClosure = new Closure_Grid::SendLoop_5_closure();
  implP|genClosure->output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->SendLoop(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_SendLoop_marshall5(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const CProxy_Grid &output*/
  PUP::fromMem implP(impl_buf);
  CProxy_Grid output; implP|output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("output");
  implDestP|output;
}
PUPable_def(SINGLE_ARG(Closure_Grid::SendLoop_5_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void pgmrun(const CkCallback &cb);
 */

void CProxy_Grid::pgmrun(const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_pgmrun_marshall6(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_pgmrun_marshall6() {
  int epidx = CkRegisterEp("pgmrun(const CkCallback &cb)",
      _call_pgmrun_marshall6, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_pgmrun_marshall6);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_pgmrun_marshall6);

  return epidx;
}


void CkIndex_Grid::_call_pgmrun_marshall6(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::pgmrun_6_closure* genClosure = new Closure_Grid::pgmrun_6_closure();
  implP|genClosure->cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->pgmrun(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_pgmrun_marshall6(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::pgmrun_6_closure* genClosure = new Closure_Grid::pgmrun_6_closure();
  implP|genClosure->cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->pgmrun(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_pgmrun_marshall6(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const CkCallback &cb*/
  PUP::fromMem implP(impl_buf);
  CkCallback cb; implP|cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("cb");
  implDestP|cb;
}
PUPable_def(SINGLE_ARG(Closure_Grid::pgmrun_6_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void pgmrunloop(const CkCallback &cb);
 */

void CProxy_Grid::pgmrunloop(const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_pgmrunloop_marshall7(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_pgmrunloop_marshall7() {
  int epidx = CkRegisterEp("pgmrunloop(const CkCallback &cb)",
      _call_pgmrunloop_marshall7, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_pgmrunloop_marshall7);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_pgmrunloop_marshall7);

  return epidx;
}


void CkIndex_Grid::_call_pgmrunloop_marshall7(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::pgmrunloop_7_closure* genClosure = new Closure_Grid::pgmrunloop_7_closure();
  implP|genClosure->cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->pgmrunloop(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_pgmrunloop_marshall7(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::pgmrunloop_7_closure* genClosure = new Closure_Grid::pgmrunloop_7_closure();
  implP|genClosure->cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->pgmrunloop(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_pgmrunloop_marshall7(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const CkCallback &cb*/
  PUP::fromMem implP(impl_buf);
  CkCallback cb; implP|cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("cb");
  implDestP|cb;
}
PUPable_def(SINGLE_ARG(Closure_Grid::pgmrunloop_7_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void postrun(const CkCallback &cb);
 */

void CProxy_Grid::postrun(const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_postrun_marshall8(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_postrun_marshall8() {
  int epidx = CkRegisterEp("postrun(const CkCallback &cb)",
      _call_postrun_marshall8, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_postrun_marshall8);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_postrun_marshall8);

  return epidx;
}


void CkIndex_Grid::_call_postrun_marshall8(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::postrun_8_closure* genClosure = new Closure_Grid::postrun_8_closure();
  implP|genClosure->cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->postrun(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_postrun_marshall8(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::postrun_8_closure* genClosure = new Closure_Grid::postrun_8_closure();
  implP|genClosure->cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->postrun(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_postrun_marshall8(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const CkCallback &cb*/
  PUP::fromMem implP(impl_buf);
  CkCallback cb; implP|cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("cb");
  implDestP|cb;
}
PUPable_def(SINGLE_ARG(Closure_Grid::postrun_8_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void input(int c_num, const unsigned char *src);
 */

void CProxy_Grid::input(int c_num, const unsigned char *src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int c_num, const unsigned char *src
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_src, impl_cnt_src;
  impl_off_src=impl_off=CK_ALIGN(impl_off,sizeof(unsigned char));
  impl_off+=(impl_cnt_src=sizeof(unsigned char)*(c_num));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_src,src,impl_cnt_src);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_input_marshall9(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_input_marshall9() {
  int epidx = CkRegisterEp("input(int c_num, const unsigned char *src)",
      _call_input_marshall9, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_input_marshall9);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_input_marshall9);

  return epidx;
}


void CkIndex_Grid::_call_input_marshall9(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::input_9_closure* genClosure = new Closure_Grid::input_9_closure();
  implP|genClosure->c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  genClosure->src = (unsigned char *)(impl_buf+impl_off_src);
  genClosure->_impl_marshall = impl_msg_typed;
  CmiReference(UsrToEnv(genClosure->_impl_marshall));
  impl_obj->input(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_input_marshall9(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::input_9_closure* genClosure = new Closure_Grid::input_9_closure();
  implP|genClosure->c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  genClosure->src = (unsigned char *)(impl_buf+impl_off_src);
  genClosure->_impl_buf_in = impl_buf;
  genClosure->_impl_buf_size = implP.size();
  impl_obj->input(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_input_marshall9(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: int c_num, const unsigned char *src*/
  PUP::fromMem implP(impl_buf);
  int c_num; implP|c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  unsigned char *src=(unsigned char *)(impl_buf+impl_off_src);
  if (implDestP.hasComments()) implDestP.comment("c_num");
  implDestP|c_num;
  if (implDestP.hasComments()) implDestP.comment("src");
  implDestP.synchronize(PUP::sync_begin_array);
  for (int impl_i=0;impl_i*(sizeof(*src))<impl_cnt_src;impl_i++) {
    implDestP.synchronize(PUP::sync_item);
    implDestP|src[impl_i];
  }
  implDestP.synchronize(PUP::sync_end_array);
}
PUPable_def(SINGLE_ARG(Closure_Grid::input_9_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void input_pos(int c_num, const float *src);
 */

void CProxy_Grid::input_pos(int c_num, const float *src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int c_num, const float *src
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_src, impl_cnt_src;
  impl_off_src=impl_off=CK_ALIGN(impl_off,sizeof(float));
  impl_off+=(impl_cnt_src=sizeof(float)*(c_num));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_src,src,impl_cnt_src);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_input_pos_marshall10(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_input_pos_marshall10() {
  int epidx = CkRegisterEp("input_pos(int c_num, const float *src)",
      _call_input_pos_marshall10, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_input_pos_marshall10);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_input_pos_marshall10);

  return epidx;
}


void CkIndex_Grid::_call_input_pos_marshall10(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::input_pos_10_closure* genClosure = new Closure_Grid::input_pos_10_closure();
  implP|genClosure->c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  genClosure->src = (float *)(impl_buf+impl_off_src);
  genClosure->_impl_marshall = impl_msg_typed;
  CmiReference(UsrToEnv(genClosure->_impl_marshall));
  impl_obj->input_pos(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_input_pos_marshall10(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::input_pos_10_closure* genClosure = new Closure_Grid::input_pos_10_closure();
  implP|genClosure->c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  genClosure->src = (float *)(impl_buf+impl_off_src);
  genClosure->_impl_buf_in = impl_buf;
  genClosure->_impl_buf_size = implP.size();
  impl_obj->input_pos(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_input_pos_marshall10(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: int c_num, const float *src*/
  PUP::fromMem implP(impl_buf);
  int c_num; implP|c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  float *src=(float *)(impl_buf+impl_off_src);
  if (implDestP.hasComments()) implDestP.comment("c_num");
  implDestP|c_num;
  if (implDestP.hasComments()) implDestP.comment("src");
  implDestP.synchronize(PUP::sync_begin_array);
  for (int impl_i=0;impl_i*(sizeof(*src))<impl_cnt_src;impl_i++) {
    implDestP.synchronize(PUP::sync_item);
    implDestP|src[impl_i];
  }
  implDestP.synchronize(PUP::sync_end_array);
}
PUPable_def(SINGLE_ARG(Closure_Grid::input_pos_10_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void update_post(CkReductionMsg* impl_msg);
 */

void CProxy_Grid::update_post(CkReductionMsg* impl_msg) 
{
  ckCheck();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_update_post_CkReductionMsg(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_update_post_CkReductionMsg() {
  int epidx = CkRegisterEp("update_post(CkReductionMsg* impl_msg)",
      _call_update_post_CkReductionMsg, CMessage_CkReductionMsg::__idx, __idx, 0);
  CkRegisterMessagePupFn(epidx, (CkMessagePupFn)CkReductionMsg::ckDebugPup);
  return epidx;
}


// Redn wrapper registration function

int CkIndex_Grid::reg_redn_wrapper_update_post_CkReductionMsg() {
  return CkRegisterEp("redn_wrapper_update_post(CkReductionMsg *impl_msg)",
      _call_redn_wrapper_update_post_CkReductionMsg, CMessage_CkReductionMsg::__idx, __idx, 0);
}


void CkIndex_Grid::_call_update_post_CkReductionMsg(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  impl_obj->update_post((CkReductionMsg*)impl_msg);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void update_res(CkReductionMsg* impl_msg);
 */

void CProxy_Grid::update_res(CkReductionMsg* impl_msg) 
{
  ckCheck();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_update_res_CkReductionMsg(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_update_res_CkReductionMsg() {
  int epidx = CkRegisterEp("update_res(CkReductionMsg* impl_msg)",
      _call_update_res_CkReductionMsg, CMessage_CkReductionMsg::__idx, __idx, 0);
  CkRegisterMessagePupFn(epidx, (CkMessagePupFn)CkReductionMsg::ckDebugPup);
  return epidx;
}


// Redn wrapper registration function

int CkIndex_Grid::reg_redn_wrapper_update_res_CkReductionMsg() {
  return CkRegisterEp("redn_wrapper_update_res(CkReductionMsg *impl_msg)",
      _call_redn_wrapper_update_res_CkReductionMsg, CMessage_CkReductionMsg::__idx, __idx, 0);
}


void CkIndex_Grid::_call_update_res_CkReductionMsg(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  impl_obj->update_res((CkReductionMsg*)impl_msg);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: Grid(CkMigrateMessage* impl_msg);
 */

// Entry point registration function

int CkIndex_Grid::reg_Grid_CkMigrateMessage() {
  int epidx = CkRegisterEp("Grid(CkMigrateMessage* impl_msg)",
      _call_Grid_CkMigrateMessage, 0, __idx, 0);
  return epidx;
}


void CkIndex_Grid::_call_Grid_CkMigrateMessage(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  new (impl_obj) Grid((CkMigrateMessage*)impl_msg);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: Grid(const bool &accept, int num_pieces_);
 */
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void cleanup();
 */

void CProxySection_Grid::cleanup(const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_cleanup_void(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void SendInput(const CProxy_Grid &output);
 */

void CProxySection_Grid::SendInput(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CProxy_Grid &output
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_SendInput_marshall3(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void SendPost(const CProxy_Grid &output);
 */

void CProxySection_Grid::SendPost(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CProxy_Grid &output
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_SendPost_marshall4(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void SendLoop(const CProxy_Grid &output);
 */

void CProxySection_Grid::SendLoop(const CProxy_Grid &output, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CProxy_Grid &output
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CProxy_Grid &)output;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_SendLoop_marshall5(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void pgmrun(const CkCallback &cb);
 */

void CProxySection_Grid::pgmrun(const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_pgmrun_marshall6(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void pgmrunloop(const CkCallback &cb);
 */

void CProxySection_Grid::pgmrunloop(const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_pgmrunloop_marshall7(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void postrun(const CkCallback &cb);
 */

void CProxySection_Grid::postrun(const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_postrun_marshall8(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void input(int c_num, const unsigned char *src);
 */

void CProxySection_Grid::input(int c_num, const unsigned char *src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int c_num, const unsigned char *src
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_src, impl_cnt_src;
  impl_off_src=impl_off=CK_ALIGN(impl_off,sizeof(unsigned char));
  impl_off+=(impl_cnt_src=sizeof(unsigned char)*(c_num));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_src,src,impl_cnt_src);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_input_marshall9(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void input_pos(int c_num, const float *src);
 */

void CProxySection_Grid::input_pos(int c_num, const float *src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int c_num, const float *src
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_src, impl_cnt_src;
  impl_off_src=impl_off=CK_ALIGN(impl_off,sizeof(float));
  impl_off+=(impl_cnt_src=sizeof(float)*(c_num));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|c_num;
    implP|impl_off_src;
    implP|impl_cnt_src;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_src,src,impl_cnt_src);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_input_pos_marshall10(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void update_post(CkReductionMsg* impl_msg);
 */

void CProxySection_Grid::update_post(CkReductionMsg* impl_msg) 
{
  ckCheck();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_update_post_CkReductionMsg(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void update_res(CkReductionMsg* impl_msg);
 */

void CProxySection_Grid::update_res(CkReductionMsg* impl_msg) 
{
  ckCheck();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_update_res_CkReductionMsg(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: Grid(CkMigrateMessage* impl_msg);
 */
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void CkIndex_Grid::__register(const char *s, size_t size) {
  __idx = CkRegisterChare(s, size, TypeArray);
  CkRegisterBase(__idx, CkIndex_ArrayElement::__idx);
  // REG: Grid(const bool &accept, int num_pieces_);
  idx_Grid_marshall1();

  // REG: void cleanup();
  idx_cleanup_void();

  // REG: void SendInput(const CProxy_Grid &output);
  idx_SendInput_marshall3();

  // REG: void SendPost(const CProxy_Grid &output);
  idx_SendPost_marshall4();

  // REG: void SendLoop(const CProxy_Grid &output);
  idx_SendLoop_marshall5();

  // REG: void pgmrun(const CkCallback &cb);
  idx_pgmrun_marshall6();

  // REG: void pgmrunloop(const CkCallback &cb);
  idx_pgmrunloop_marshall7();

  // REG: void postrun(const CkCallback &cb);
  idx_postrun_marshall8();

  // REG: void input(int c_num, const unsigned char *src);
  idx_input_marshall9();

  // REG: void input_pos(int c_num, const float *src);
  idx_input_pos_marshall10();

  // REG: void update_post(CkReductionMsg* impl_msg);
  idx_update_post_CkReductionMsg();
  idx_redn_wrapper_update_post_CkReductionMsg();

  // REG: void update_res(CkReductionMsg* impl_msg);
  idx_update_res_CkReductionMsg();
  idx_redn_wrapper_update_res_CkReductionMsg();

  // REG: Grid(CkMigrateMessage* impl_msg);
  idx_Grid_CkMigrateMessage();
  CkRegisterMigCtor(__idx, idx_Grid_CkMigrateMessage());

  Grid::__sdag_register(); // Potentially missing Grid_SDAG_CODE in your class definition?
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
void Grid::SendInput(CProxy_Grid output){
  Closure_Grid::SendInput_3_closure* genClosure = new Closure_Grid::SendInput_3_closure();
  genClosure->getP0() = output;
  SendInput(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::SendInput(Closure_Grid::SendInput_3_closure* gen0) {
  _TRACE_END_EXECUTE(); 
  if (!__dep.get()) _sdag_init();
  _slist_0(gen0);
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::SendInput_end(Closure_Grid::SendInput_3_closure* gen0) {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_0(Closure_Grid::SendInput_3_closure* gen0) {
  _atomic_0(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_0_end(Closure_Grid::SendInput_3_closure* gen0) {
  SendInput_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_atomic_0(Closure_Grid::SendInput_3_closure* gen0) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Grid_atomic_0()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    CProxy_Grid& output = gen0->getP0();
    { // begin serial block
#line 21 "circuit.ci"

                output.input(mem_size, mem_begin);
            
#line 2287 "circuit.def.h"
    } // end serial block
  }
  _TRACE_END_EXECUTE(); 
  _slist_0_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::SendPost(CProxy_Grid output){
  Closure_Grid::SendPost_4_closure* genClosure = new Closure_Grid::SendPost_4_closure();
  genClosure->getP0() = output;
  SendPost(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::SendPost(Closure_Grid::SendPost_4_closure* gen0) {
  _TRACE_END_EXECUTE(); 
  if (!__dep.get()) _sdag_init();
  _slist_1(gen0);
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::SendPost_end(Closure_Grid::SendPost_4_closure* gen0) {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_1(Closure_Grid::SendPost_4_closure* gen0) {
  _atomic_1(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_1_end(Closure_Grid::SendPost_4_closure* gen0) {
  SendPost_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_atomic_1(Closure_Grid::SendPost_4_closure* gen0) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Grid_atomic_1()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    CProxy_Grid& output = gen0->getP0();
    { // begin serial block
#line 26 "circuit.ci"

                CkPrintf("Rank: %d, SendPost--->\n", thisIndex);
                output.input_pos(post_num, post_charge);
                CkPrintf("SendPost complete--->");
            
#line 2347 "circuit.def.h"
    } // end serial block
  }
  _TRACE_END_EXECUTE(); 
  _slist_1_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::SendLoop(CProxy_Grid output){
  Closure_Grid::SendLoop_5_closure* genClosure = new Closure_Grid::SendLoop_5_closure();
  genClosure->getP0() = output;
  SendLoop(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::SendLoop(Closure_Grid::SendLoop_5_closure* gen0) {
  _TRACE_END_EXECUTE(); 
  if (!__dep.get()) _sdag_init();
  _slist_2(gen0);
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::SendLoop_end(Closure_Grid::SendLoop_5_closure* gen0) {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_2(Closure_Grid::SendLoop_5_closure* gen0) {
  _atomic_2(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_2_end(Closure_Grid::SendLoop_5_closure* gen0) {
  SendLoop_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_atomic_2(Closure_Grid::SendLoop_5_closure* gen0) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Grid_atomic_2()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    CProxy_Grid& output = gen0->getP0();
    { // begin serial block
#line 33 "circuit.ci"

                CkPrintf("Rank: %d, SendLoop--->\n", thisIndex);
                output.input_pos(post_shr_num, post_shr_voltage);
                CkPrintf("SendLoop complete--->");
            
#line 2407 "circuit.def.h"
    } // end serial block
  }
  _TRACE_END_EXECUTE(); 
  _slist_2_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::pgmrun(CkCallback & cb){
  Closure_Grid::pgmrun_6_closure* genClosure = new Closure_Grid::pgmrun_6_closure();
  genClosure->getP0() = cb;
  pgmrun(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::pgmrun(Closure_Grid::pgmrun_6_closure* gen0) {
  _TRACE_END_EXECUTE(); 
  if (!__dep.get()) _sdag_init();
  _slist_3(gen0);
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::pgmrun_end(Closure_Grid::pgmrun_6_closure* gen0) {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_3(Closure_Grid::pgmrun_6_closure* gen0) {
  _when_0(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_3_end(Closure_Grid::pgmrun_6_closure* gen0) {
  pgmrun_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
SDAG::Continuation* Grid::_when_0(Closure_Grid::pgmrun_6_closure* gen0) {
  SDAG::Buffer* buf0 = __dep->tryFindMessage(0, false, 0, 0);
  if (buf0) {
    __dep->removeMessage(buf0);
    _atomic_3(gen0, static_cast<Closure_Grid::input_9_closure*>(buf0->cl));
    delete buf0;
    return 0;
  } else {
    SDAG::Continuation* c = new SDAG::Continuation(0);
    c->addClosure(gen0);
    c->anyEntries.push_back(0);
    __dep->reg(c);
    return c;
  }
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_when_0_end(Closure_Grid::pgmrun_6_closure* gen0, Closure_Grid::input_9_closure* gen1) {
  _slist_3_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_atomic_3(Closure_Grid::pgmrun_6_closure* gen0, Closure_Grid::input_9_closure* gen1) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Grid_atomic_3()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    CkCallback& cb = gen0->getP0();
    {
      int& c_num = gen1->getP0();
      unsigned char*& src = gen1->getP1();
      { // begin serial block
#line 42 "circuit.ci"

                init_val(num_pieces*mem_pc_size, mem_begin, (src+thisIndex*num_pieces*mem_pc_size));
                cudaInit(true, node_piece, wire_piece, transfer_buf, nodes_per_piece, wires_per_piece, num_pieces, thisIndex, num_blocks, num_threads);
                contribute(transfer_size, transfer_buf, CkReduction::set, cb);
            
#line 2496 "circuit.def.h"
      } // end serial block
    }
  }
  _TRACE_END_EXECUTE(); 
  _when_0_end(gen0, gen1);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::pgmrunloop(CkCallback & cb){
  Closure_Grid::pgmrunloop_7_closure* genClosure = new Closure_Grid::pgmrunloop_7_closure();
  genClosure->getP0() = cb;
  pgmrunloop(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::pgmrunloop(Closure_Grid::pgmrunloop_7_closure* gen0) {
  _TRACE_END_EXECUTE(); 
  if (!__dep.get()) _sdag_init();
  _slist_4(gen0);
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::pgmrunloop_end(Closure_Grid::pgmrunloop_7_closure* gen0) {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_4(Closure_Grid::pgmrunloop_7_closure* gen0) {
  _when_1(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_4_end(Closure_Grid::pgmrunloop_7_closure* gen0) {
  pgmrunloop_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
SDAG::Continuation* Grid::_when_1(Closure_Grid::pgmrunloop_7_closure* gen0) {
  SDAG::Buffer* buf0 = __dep->tryFindMessage(1, false, 0, 0);
  if (buf0) {
    __dep->removeMessage(buf0);
    _atomic_4(gen0, static_cast<Closure_Grid::input_pos_10_closure*>(buf0->cl));
    delete buf0;
    return 0;
  } else {
    SDAG::Continuation* c = new SDAG::Continuation(1);
    c->addClosure(gen0);
    c->anyEntries.push_back(1);
    __dep->reg(c);
    return c;
  }
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_when_1_end(Closure_Grid::pgmrunloop_7_closure* gen0, Closure_Grid::input_pos_10_closure* gen1) {
  _slist_4_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_atomic_4(Closure_Grid::pgmrunloop_7_closure* gen0, Closure_Grid::input_pos_10_closure* gen1) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Grid_atomic_4()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    CkCallback& cb = gen0->getP0();
    {
      int& c_num = gen1->getP0();
      float*& src = gen1->getP1();
      { // begin serial block
#line 51 "circuit.ci"

                init_post_shr(wire_piece, src, thisIndex, wires_per_piece, num_pieces);
                cudaInit(true, node_piece, wire_piece, transfer_buf, nodes_per_piece, wires_per_piece, num_pieces, thisIndex, num_blocks, num_threads);
                contribute(transfer_size, transfer_buf, CkReduction::set, cb);
            
#line 2586 "circuit.def.h"
      } // end serial block
    }
  }
  _TRACE_END_EXECUTE(); 
  _when_1_end(gen0, gen1);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::postrun(CkCallback & cb){
  Closure_Grid::postrun_8_closure* genClosure = new Closure_Grid::postrun_8_closure();
  genClosure->getP0() = cb;
  postrun(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::postrun(Closure_Grid::postrun_8_closure* gen0) {
  _TRACE_END_EXECUTE(); 
  if (!__dep.get()) _sdag_init();
  _slist_5(gen0);
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::postrun_end(Closure_Grid::postrun_8_closure* gen0) {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_5(Closure_Grid::postrun_8_closure* gen0) {
  _when_2(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_5_end(Closure_Grid::postrun_8_closure* gen0) {
  postrun_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
SDAG::Continuation* Grid::_when_2(Closure_Grid::postrun_8_closure* gen0) {
  SDAG::Buffer* buf0 = __dep->tryFindMessage(1, false, 0, 0);
  if (buf0) {
    __dep->removeMessage(buf0);
    _atomic_5(gen0, static_cast<Closure_Grid::input_pos_10_closure*>(buf0->cl));
    delete buf0;
    return 0;
  } else {
    SDAG::Continuation* c = new SDAG::Continuation(2);
    c->addClosure(gen0);
    c->anyEntries.push_back(1);
    __dep->reg(c);
    return c;
  }
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_when_2_end(Closure_Grid::postrun_8_closure* gen0, Closure_Grid::input_pos_10_closure* gen1) {
  _slist_5_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_atomic_5(Closure_Grid::postrun_8_closure* gen0, Closure_Grid::input_pos_10_closure* gen1) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Grid_atomic_5()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    CkCallback& cb = gen0->getP0();
    {
      int& c_num = gen1->getP0();
      float*& src = gen1->getP1();
      { // begin serial block
#line 60 "circuit.ci"

                CkPrintf("My Rank:%d, postrun--->\n", thisIndex);
                for (int i=0 ; i<num_pieces; i++) {
                    for (int j=0; j<wires_per_piece; j++) {
                        wire_piece[i].shr_charge[j] = 0.f;
                    }
                }
                int post_per_pe_size =nodes_per_piece * sizeof(float);
                init_post(node_piece, src, thisIndex, nodes_per_piece, num_pieces);
                cudaInit(false, node_piece, wire_piece, result_buf, nodes_per_piece, wires_per_piece, num_pieces, thisIndex, num_blocks, num_threads);
                contribute(result_size, result_buf, CkReduction::set, cb);
            
#line 2683 "circuit.def.h"
      } // end serial block
    }
  }
  _TRACE_END_EXECUTE(); 
  _when_2_end(gen0, gen1);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::input(int c_num, unsigned char *src){
  Closure_Grid::input_9_closure* genClosure = new Closure_Grid::input_9_closure();
  genClosure->getP0() = c_num;
  genClosure->getP1() = src;
  input(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::input(Closure_Grid::input_9_closure* genClosure){
  if (!__dep.get()) _sdag_init();
  __dep->pushBuffer(0, genClosure, 0);
  SDAG::Continuation* c = __dep->tryFindContinuation(0);
  if (c) {
    _TRACE_END_EXECUTE(); 
    _when_0(
      static_cast<Closure_Grid::pgmrun_6_closure*>(c->closure[0])
    );
    _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
    delete c;
  }
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::input_pos(int c_num, float *src){
  Closure_Grid::input_pos_10_closure* genClosure = new Closure_Grid::input_pos_10_closure();
  genClosure->getP0() = c_num;
  genClosure->getP1() = src;
  input_pos(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::input_pos(Closure_Grid::input_pos_10_closure* genClosure){
  if (!__dep.get()) _sdag_init();
  __dep->pushBuffer(1, genClosure, 0);
  SDAG::Continuation* c = __dep->tryFindContinuation(1);
  if (c) {
    _TRACE_END_EXECUTE(); 
    switch(c->whenID) {
    case 1:
      _when_1(
        static_cast<Closure_Grid::pgmrunloop_7_closure*>(c->closure[0])
      );
    break;
    case 2:
      _when_2(
        static_cast<Closure_Grid::postrun_8_closure*>(c->closure[0])
      );
    break;
    }
    _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
    delete c;
  }
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::_sdag_init() { // Potentially missing Grid_SDAG_CODE in your class definition?
  __dep.reset(new SDAG::Dependency(2,3));
  __dep->addDepends(0,0);
  __dep->addDepends(1,1);
  __dep->addDepends(2,1);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::__sdag_init() { // Potentially missing Grid_SDAG_CODE in your class definition?
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_sdag_pup(PUP::er &p) { // Potentially missing Grid_SDAG_CODE in your class definition?
#if CMK_USING_XLC
    bool hasSDAG = __dep.get();
    p|hasSDAG;
    if (p.isUnpacking() && hasSDAG) _sdag_init();
    if (hasSDAG) { __dep->pup(p); }
#else
    p|__dep;
#endif
}
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::__sdag_register() { // Potentially missing Grid_SDAG_CODE in your class definition?
  (void)_sdag_idx_Grid_atomic_0();
  (void)_sdag_idx_Grid_atomic_1();
  (void)_sdag_idx_Grid_atomic_2();
  (void)_sdag_idx_Grid_atomic_3();
  (void)_sdag_idx_Grid_atomic_4();
  (void)_sdag_idx_Grid_atomic_5();
  PUPable_reg(SINGLE_ARG(Closure_Grid::cleanup_2_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::SendInput_3_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::SendPost_4_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::SendLoop_5_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::pgmrun_6_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::pgmrunloop_7_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::postrun_8_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::input_9_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::input_pos_10_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::cleanup_2_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::SendInput_3_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::SendPost_4_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::SendLoop_5_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::pgmrun_6_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::pgmrunloop_7_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::postrun_8_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::input_9_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::input_pos_10_closure));
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_idx_Grid_atomic_0() { // Potentially missing Grid_SDAG_CODE in your class definition?
  static int epidx = _sdag_reg_Grid_atomic_0();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_reg_Grid_atomic_0() { // Potentially missing Grid_SDAG_CODE in your class definition?
  return CkRegisterEp("Grid_atomic_0", NULL, 0, CkIndex_Grid::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_idx_Grid_atomic_1() { // Potentially missing Grid_SDAG_CODE in your class definition?
  static int epidx = _sdag_reg_Grid_atomic_1();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_reg_Grid_atomic_1() { // Potentially missing Grid_SDAG_CODE in your class definition?
  return CkRegisterEp("Grid_atomic_1", NULL, 0, CkIndex_Grid::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_idx_Grid_atomic_2() { // Potentially missing Grid_SDAG_CODE in your class definition?
  static int epidx = _sdag_reg_Grid_atomic_2();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_reg_Grid_atomic_2() { // Potentially missing Grid_SDAG_CODE in your class definition?
  return CkRegisterEp("Grid_atomic_2", NULL, 0, CkIndex_Grid::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_idx_Grid_atomic_3() { // Potentially missing Grid_SDAG_CODE in your class definition?
  static int epidx = _sdag_reg_Grid_atomic_3();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_reg_Grid_atomic_3() { // Potentially missing Grid_SDAG_CODE in your class definition?
  return CkRegisterEp("Grid_atomic_3", NULL, 0, CkIndex_Grid::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_idx_Grid_atomic_4() { // Potentially missing Grid_SDAG_CODE in your class definition?
  static int epidx = _sdag_reg_Grid_atomic_4();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_reg_Grid_atomic_4() { // Potentially missing Grid_SDAG_CODE in your class definition?
  return CkRegisterEp("Grid_atomic_4", NULL, 0, CkIndex_Grid::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_idx_Grid_atomic_5() { // Potentially missing Grid_SDAG_CODE in your class definition?
  static int epidx = _sdag_reg_Grid_atomic_5();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_reg_Grid_atomic_5() { // Potentially missing Grid_SDAG_CODE in your class definition?
  return CkRegisterEp("Grid_atomic_5", NULL, 0, CkIndex_Grid::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */



#ifndef CK_TEMPLATES_ONLY
void _registercircuit(void)
{
  static int _done = 0; if(_done) return; _done = 1;
  CkRegisterReadonly("mainProxy","CProxy_Main",sizeof(mainProxy),(void *) &mainProxy,__xlater_roPup_mainProxy);

  CkRegisterReadonly("num_pieces","int",sizeof(num_pieces),(void *) &num_pieces,__xlater_roPup_num_pieces);

  CkRegisterReadonly("max_pe","int",sizeof(max_pe),(void *) &max_pe,__xlater_roPup_max_pe);

  CkRegisterReadonly("number_loops","int",sizeof(number_loops),(void *) &number_loops,__xlater_roPup_number_loops);

  CkRegisterReadonly("nodes_per_piece","int",sizeof(nodes_per_piece),(void *) &nodes_per_piece,__xlater_roPup_nodes_per_piece);

  CkRegisterReadonly("wires_per_piece","int",sizeof(wires_per_piece),(void *) &wires_per_piece,__xlater_roPup_wires_per_piece);

  CkRegisterReadonly("pct_wire_in_piece","int",sizeof(pct_wire_in_piece),(void *) &pct_wire_in_piece,__xlater_roPup_pct_wire_in_piece);

  CkRegisterReadonly("random_seed","int",sizeof(random_seed),(void *) &random_seed,__xlater_roPup_random_seed);

  CkRegisterReadonly("num_blocks","int",sizeof(num_blocks),(void *) &num_blocks,__xlater_roPup_num_blocks);

  CkRegisterReadonly("num_threads","int",sizeof(num_threads),(void *) &num_threads,__xlater_roPup_num_threads);

/* REG: mainchare Main: Chare{
Main(CkArgMsg* impl_msg);
void done();
void post_run();
};
*/
  CkIndex_Main::__register("Main", sizeof(Main));

/* REG: array Grid: ArrayElement{
Grid(const bool &accept, int num_pieces_);
void cleanup();
void SendInput(const CProxy_Grid &output);
void SendPost(const CProxy_Grid &output);
void SendLoop(const CProxy_Grid &output);
void pgmrun(const CkCallback &cb);
void pgmrunloop(const CkCallback &cb);
void postrun(const CkCallback &cb);
void input(int c_num, const unsigned char *src);
void input_pos(int c_num, const float *src);
void update_post(CkReductionMsg* impl_msg);
void update_res(CkReductionMsg* impl_msg);
Grid(CkMigrateMessage* impl_msg);
};
*/
  CkIndex_Grid::__register("Grid", sizeof(Grid));

}
extern "C" void CkRegisterMainModule(void) {
  _registercircuit();
}
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
template <>
void CBase_Main::virtual_pup(PUP::er &p) {
    recursive_pup<Main >(dynamic_cast<Main* >(this), p);
}
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
template <>
void CBase_Grid::virtual_pup(PUP::er &p) {
    recursive_pup<Grid >(dynamic_cast<Grid* >(this), p);
}
#endif /* CK_TEMPLATES_ONLY */
