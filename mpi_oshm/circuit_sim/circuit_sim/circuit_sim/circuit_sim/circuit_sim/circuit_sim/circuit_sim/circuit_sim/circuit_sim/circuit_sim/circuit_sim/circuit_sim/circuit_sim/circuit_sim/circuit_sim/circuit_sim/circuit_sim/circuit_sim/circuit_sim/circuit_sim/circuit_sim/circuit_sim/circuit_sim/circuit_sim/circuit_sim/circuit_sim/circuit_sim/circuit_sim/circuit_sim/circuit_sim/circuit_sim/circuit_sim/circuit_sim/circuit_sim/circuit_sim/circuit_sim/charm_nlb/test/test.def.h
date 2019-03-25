
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


/* ---------------- method closures -------------- */
#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::SendInput_2_closure : public SDAG::Closure {
      CProxy_Grid output;


      SendInput_2_closure() {
        init();
      }
      SendInput_2_closure(CkMigrateMessage*) {
        init();
      }
      CProxy_Grid & getP0() { return output;}
      void pup(PUP::er& __p) {
        __p | output;
        packClosure(__p);
      }
      virtual ~SendInput_2_closure() {
      }
      PUPable_decl(SINGLE_ARG(SendInput_2_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::printout_3_closure : public SDAG::Closure {
      int num;
      CkCallback cb;


      printout_3_closure() {
        init();
      }
      printout_3_closure(CkMigrateMessage*) {
        init();
      }
      int & getP0() { return num;}
      CkCallback & getP1() { return cb;}
      void pup(PUP::er& __p) {
        __p | num;
        __p | cb;
        packClosure(__p);
      }
      virtual ~printout_3_closure() {
      }
      PUPable_decl(SINGLE_ARG(printout_3_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::printmain_4_closure : public SDAG::Closure {
      int num;


      printmain_4_closure() {
        init();
      }
      printmain_4_closure(CkMigrateMessage*) {
        init();
      }
      int & getP0() { return num;}
      void pup(PUP::er& __p) {
        __p | num;
        packClosure(__p);
      }
      virtual ~printmain_4_closure() {
      }
      PUPable_decl(SINGLE_ARG(printmain_4_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Grid::input_5_closure : public SDAG::Closure {
      int c_num;
      int *src;

      CkMarshallMsg* _impl_marshall;
      char* _impl_buf_in;
      int _impl_buf_size;

      input_5_closure() {
        init();
        _impl_marshall = 0;
        _impl_buf_in = 0;
        _impl_buf_size = 0;
      }
      input_5_closure(CkMigrateMessage*) {
        init();
        _impl_marshall = 0;
        _impl_buf_in = 0;
        _impl_buf_size = 0;
      }
      int & getP0() { return c_num;}
      int *& getP1() { return src;}
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
          src = (int *)(impl_buf+impl_off_src);
        }
      }
      virtual ~input_5_closure() {
        if (_impl_marshall) CmiFree(UsrToEnv(_impl_marshall));
      }
      PUPable_decl(SINGLE_ARG(input_5_closure));
    };
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

/* DEFS: mainchare Main: Chare{
Main(CkArgMsg* impl_msg);
void done();
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

}
#endif /* CK_TEMPLATES_ONLY */

/* DEFS: array Grid: ArrayElement{
Grid(const bool &accept, int work_num);
void SendInput(const CProxy_Grid &output);
void printout(int num, const CkCallback &cb);
void printmain(int num);
void input(int c_num, const int *src);
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
/* DEFS: Grid(const bool &accept, int work_num);
 */

void CProxyElement_Grid::insert(const bool &accept, int work_num, int onPE, const CkEntryOptions *impl_e_opts)
{ 
  //Marshall: const bool &accept, int work_num
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|work_num;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|work_num;
  }
   UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
   ckInsert((CkArrayMessage *)impl_msg,CkIndex_Grid::idx_Grid_marshall1(),onPE);
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
  ckSend(impl_amsg, CkIndex_Grid::idx_SendInput_marshall2(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void printout(int num, const CkCallback &cb);
 */

void CProxyElement_Grid::printout(int num, const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int num, const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|num;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|num;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_printout_marshall3(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void printmain(int num);
 */

void CProxyElement_Grid::printmain(int num, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int num
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|num;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|num;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_printmain_marshall4(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void input(int c_num, const int *src);
 */

void CProxyElement_Grid::input(int c_num, const int *src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int c_num, const int *src
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_src, impl_cnt_src;
  impl_off_src=impl_off=CK_ALIGN(impl_off,sizeof(int));
  impl_off+=(impl_cnt_src=sizeof(int)*(c_num));
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
  ckSend(impl_amsg, CkIndex_Grid::idx_input_marshall5(),0);
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
/* DEFS: Grid(const bool &accept, int work_num);
 */

CkArrayID CProxy_Grid::ckNew(const bool &accept, int work_num, const CkArrayOptions &opts, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const bool &accept, int work_num
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|work_num;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|work_num;
  }
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkArrayID gId = ckCreateArray((CkArrayMessage *)impl_msg, CkIndex_Grid::idx_Grid_marshall1(), opts);
  return gId;
}

void CProxy_Grid::ckNew(const bool &accept, int work_num, const CkArrayOptions &opts, CkCallback _ck_array_creation_cb, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const bool &accept, int work_num
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|work_num;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|work_num;
  }
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkSendAsyncCreateArray(CkIndex_Grid::idx_Grid_marshall1(), _ck_array_creation_cb, opts, impl_msg);
}

CkArrayID CProxy_Grid::ckNew(const bool &accept, int work_num, const int s1, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const bool &accept, int work_num
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|work_num;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|work_num;
  }
  CkArrayOptions opts(s1);
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkArrayID gId = ckCreateArray((CkArrayMessage *)impl_msg, CkIndex_Grid::idx_Grid_marshall1(), opts);
  return gId;
}

void CProxy_Grid::ckNew(const bool &accept, int work_num, const int s1, CkCallback _ck_array_creation_cb, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const bool &accept, int work_num
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|work_num;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    //Have to cast away const-ness to get pup routine
    implP|(bool &)accept;
    implP|work_num;
  }
  CkArrayOptions opts(s1);
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkSendAsyncCreateArray(CkIndex_Grid::idx_Grid_marshall1(), _ck_array_creation_cb, opts, impl_msg);
}

// Entry point registration function

int CkIndex_Grid::reg_Grid_marshall1() {
  int epidx = CkRegisterEp("Grid(const bool &accept, int work_num)",
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
  /*Unmarshall pup'd fields: const bool &accept, int work_num*/
  PUP::fromMem implP(impl_buf);
  bool accept; implP|accept;
  int work_num; implP|work_num;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  new (impl_obj) Grid(accept, work_num);
}

int CkIndex_Grid::_callmarshall_Grid_marshall1(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  /*Unmarshall pup'd fields: const bool &accept, int work_num*/
  PUP::fromMem implP(impl_buf);
  bool accept; implP|accept;
  int work_num; implP|work_num;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  new (impl_obj) Grid(accept, work_num);
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_Grid_marshall1(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const bool &accept, int work_num*/
  PUP::fromMem implP(impl_buf);
  bool accept; implP|accept;
  int work_num; implP|work_num;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("accept");
  implDestP|accept;
  if (implDestP.hasComments()) implDestP.comment("work_num");
  implDestP|work_num;
}
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
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_SendInput_marshall2(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_SendInput_marshall2() {
  int epidx = CkRegisterEp("SendInput(const CProxy_Grid &output)",
      _call_SendInput_marshall2, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_SendInput_marshall2);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_SendInput_marshall2);

  return epidx;
}


void CkIndex_Grid::_call_SendInput_marshall2(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::SendInput_2_closure* genClosure = new Closure_Grid::SendInput_2_closure();
  implP|genClosure->output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->SendInput(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_SendInput_marshall2(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::SendInput_2_closure* genClosure = new Closure_Grid::SendInput_2_closure();
  implP|genClosure->output;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->SendInput(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_SendInput_marshall2(PUP::er &implDestP,void *impl_msg) {
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
PUPable_def(SINGLE_ARG(Closure_Grid::SendInput_2_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void printout(int num, const CkCallback &cb);
 */

void CProxy_Grid::printout(int num, const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int num, const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|num;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|num;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_printout_marshall3(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_printout_marshall3() {
  int epidx = CkRegisterEp("printout(int num, const CkCallback &cb)",
      _call_printout_marshall3, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_printout_marshall3);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_printout_marshall3);

  return epidx;
}


void CkIndex_Grid::_call_printout_marshall3(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::printout_3_closure* genClosure = new Closure_Grid::printout_3_closure();
  implP|genClosure->num;
  implP|genClosure->cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->printout(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_printout_marshall3(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::printout_3_closure* genClosure = new Closure_Grid::printout_3_closure();
  implP|genClosure->num;
  implP|genClosure->cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->printout(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_printout_marshall3(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: int num, const CkCallback &cb*/
  PUP::fromMem implP(impl_buf);
  int num; implP|num;
  CkCallback cb; implP|cb;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("num");
  implDestP|num;
  if (implDestP.hasComments()) implDestP.comment("cb");
  implDestP|cb;
}
PUPable_def(SINGLE_ARG(Closure_Grid::printout_3_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void printmain(int num);
 */

void CProxy_Grid::printmain(int num, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int num
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|num;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|num;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_printmain_marshall4(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_printmain_marshall4() {
  int epidx = CkRegisterEp("printmain(int num)",
      _call_printmain_marshall4, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_printmain_marshall4);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_printmain_marshall4);

  return epidx;
}


void CkIndex_Grid::_call_printmain_marshall4(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::printmain_4_closure* genClosure = new Closure_Grid::printmain_4_closure();
  implP|genClosure->num;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->printmain(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_printmain_marshall4(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::printmain_4_closure* genClosure = new Closure_Grid::printmain_4_closure();
  implP|genClosure->num;
  impl_buf+=CK_ALIGN(implP.size(),16);
  impl_obj->printmain(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_printmain_marshall4(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: int num*/
  PUP::fromMem implP(impl_buf);
  int num; implP|num;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  if (implDestP.hasComments()) implDestP.comment("num");
  implDestP|num;
}
PUPable_def(SINGLE_ARG(Closure_Grid::printmain_4_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void input(int c_num, const int *src);
 */

void CProxy_Grid::input(int c_num, const int *src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int c_num, const int *src
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_src, impl_cnt_src;
  impl_off_src=impl_off=CK_ALIGN(impl_off,sizeof(int));
  impl_off+=(impl_cnt_src=sizeof(int)*(c_num));
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
  ckBroadcast(impl_amsg, CkIndex_Grid::idx_input_marshall5(),0);
}

// Entry point registration function

int CkIndex_Grid::reg_input_marshall5() {
  int epidx = CkRegisterEp("input(int c_num, const int *src)",
      _call_input_marshall5, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_input_marshall5);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_input_marshall5);

  return epidx;
}


void CkIndex_Grid::_call_input_marshall5(void* impl_msg, void* impl_obj_void)
{
  Grid* impl_obj = static_cast<Grid *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Grid::input_5_closure* genClosure = new Closure_Grid::input_5_closure();
  implP|genClosure->c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  genClosure->src = (int *)(impl_buf+impl_off_src);
  genClosure->_impl_marshall = impl_msg_typed;
  CmiReference(UsrToEnv(genClosure->_impl_marshall));
  impl_obj->input(genClosure);
  genClosure->deref();
}

int CkIndex_Grid::_callmarshall_input_marshall5(char* impl_buf, void* impl_obj_void) {
  Grid* impl_obj = static_cast< Grid *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Grid::input_5_closure* genClosure = new Closure_Grid::input_5_closure();
  implP|genClosure->c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  genClosure->src = (int *)(impl_buf+impl_off_src);
  genClosure->_impl_buf_in = impl_buf;
  genClosure->_impl_buf_size = implP.size();
  impl_obj->input(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Grid::_marshallmessagepup_input_marshall5(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: int c_num, const int *src*/
  PUP::fromMem implP(impl_buf);
  int c_num; implP|c_num;
  int impl_off_src, impl_cnt_src; 
  implP|impl_off_src;
  implP|impl_cnt_src;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  int *src=(int *)(impl_buf+impl_off_src);
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
PUPable_def(SINGLE_ARG(Closure_Grid::input_5_closure))
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
/* DEFS: Grid(const bool &accept, int work_num);
 */
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
  ckSend(impl_amsg, CkIndex_Grid::idx_SendInput_marshall2(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void printout(int num, const CkCallback &cb);
 */

void CProxySection_Grid::printout(int num, const CkCallback &cb, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int num, const CkCallback &cb
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|num;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|num;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)cb;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_printout_marshall3(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void printmain(int num);
 */

void CProxySection_Grid::printmain(int num, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int num
  int impl_off=0;
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|num;
    impl_off+=implP.size();
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|num;
  }
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Grid::idx_printmain_marshall4(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void input(int c_num, const int *src);
 */

void CProxySection_Grid::input(int c_num, const int *src, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: int c_num, const int *src
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_src, impl_cnt_src;
  impl_off_src=impl_off=CK_ALIGN(impl_off,sizeof(int));
  impl_off+=(impl_cnt_src=sizeof(int)*(c_num));
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
  ckSend(impl_amsg, CkIndex_Grid::idx_input_marshall5(),0);
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
  // REG: Grid(const bool &accept, int work_num);
  idx_Grid_marshall1();

  // REG: void SendInput(const CProxy_Grid &output);
  idx_SendInput_marshall2();

  // REG: void printout(int num, const CkCallback &cb);
  idx_printout_marshall3();

  // REG: void printmain(int num);
  idx_printmain_marshall4();

  // REG: void input(int c_num, const int *src);
  idx_input_marshall5();

  // REG: void update_res(CkReductionMsg* impl_msg);
  idx_update_res_CkReductionMsg();
  idx_redn_wrapper_update_res_CkReductionMsg();

  // REG: Grid(CkMigrateMessage* impl_msg);
  idx_Grid_CkMigrateMessage();
  CkRegisterMigCtor(__idx, idx_Grid_CkMigrateMessage());

  Grid::__sdag_register(); 
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
void Grid::SendInput(CProxy_Grid output){
  Closure_Grid::SendInput_2_closure* genClosure = new Closure_Grid::SendInput_2_closure();
  genClosure->getP0() = output;
  SendInput(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::SendInput(Closure_Grid::SendInput_2_closure* gen0) {
  _TRACE_END_EXECUTE(); 
  if (!__dep.get()) _sdag_init();
  _slist_0(gen0);
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::SendInput_end(Closure_Grid::SendInput_2_closure* gen0) {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_0(Closure_Grid::SendInput_2_closure* gen0) {
  _atomic_0(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_0_end(Closure_Grid::SendInput_2_closure* gen0) {
  SendInput_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_atomic_0(Closure_Grid::SendInput_2_closure* gen0) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Grid_atomic_0()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    CProxy_Grid& output = gen0->getP0();
    { // begin serial block
#line 10 "STDIN"

                output.input(number, data);
            
#line 1254 "test.def.h"
    } // end serial block
  }
  _TRACE_END_EXECUTE(); 
  _slist_0_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::printout(int num, CkCallback & cb){
  Closure_Grid::printout_3_closure* genClosure = new Closure_Grid::printout_3_closure();
  genClosure->getP0() = num;
  genClosure->getP1() = cb;
  printout(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::printout(Closure_Grid::printout_3_closure* gen0) {
  _TRACE_END_EXECUTE(); 
  if (!__dep.get()) _sdag_init();
  _slist_1(gen0);
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::printout_end(Closure_Grid::printout_3_closure* gen0) {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_1(Closure_Grid::printout_3_closure* gen0) {
  _when_0(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_1_end(Closure_Grid::printout_3_closure* gen0) {
  printout_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
SDAG::Continuation* Grid::_when_0(Closure_Grid::printout_3_closure* gen0) {
  SDAG::Buffer* buf0 = __dep->tryFindMessage(0, false, 0, 0);
  if (buf0) {
    __dep->removeMessage(buf0);
    _atomic_1(gen0, static_cast<Closure_Grid::input_5_closure*>(buf0->cl));
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
void Grid::_when_0_end(Closure_Grid::printout_3_closure* gen0, Closure_Grid::input_5_closure* gen1) {
  _atomic_2(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_atomic_1(Closure_Grid::printout_3_closure* gen0, Closure_Grid::input_5_closure* gen1) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Grid_atomic_1()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    int& num = gen0->getP0();
    CkCallback& cb = gen0->getP1();
    {
      int& c_num = gen1->getP0();
      int*& src = gen1->getP1();
      { // begin serial block
#line 17 "STDIN"

                init_val(2, data, (src+thisIndex*2));
                CkPrintf("My rank: %d\n", thisIndex);
                for (int i=0; i<num; i++)
                    CkPrintf("%d ", data[i]);
                CkPrintf("\n");
            
#line 1347 "test.def.h"
      } // end serial block
    }
  }
  _TRACE_END_EXECUTE(); 
  _when_0_end(gen0, gen1);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_atomic_2(Closure_Grid::printout_3_closure* gen0) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Grid_atomic_2()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    int& num = gen0->getP0();
    CkCallback& cb = gen0->getP1();
    { // begin serial block
#line 24 "STDIN"

                contribute(2*sizeof(int), data, CkReduction::set, cb);
            
#line 1368 "test.def.h"
    } // end serial block
  }
  _TRACE_END_EXECUTE(); 
  _slist_1_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::printmain(int num){
  Closure_Grid::printmain_4_closure* genClosure = new Closure_Grid::printmain_4_closure();
  genClosure->getP0() = num;
  printmain(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::printmain(Closure_Grid::printmain_4_closure* gen0) {
  _TRACE_END_EXECUTE(); 
  if (!__dep.get()) _sdag_init();
  _slist_2(gen0);
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::printmain_end(Closure_Grid::printmain_4_closure* gen0) {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_2(Closure_Grid::printmain_4_closure* gen0) {
  _atomic_3(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_slist_2_end(Closure_Grid::printmain_4_closure* gen0) {
  printmain_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_atomic_3(Closure_Grid::printmain_4_closure* gen0) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Grid_atomic_3()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    int& num = gen0->getP0();
    { // begin serial block
#line 29 "STDIN"

                CkPrintf("My rank: %d\n", thisIndex);
                for (int i=0; i<num; i++)
                    CkPrintf("%d ", data[i]);
                CkPrintf("\n");
                mainProxy.done();
            
#line 1430 "test.def.h"
    } // end serial block
  }
  _TRACE_END_EXECUTE(); 
  _slist_2_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::input(int c_num, int *src){
  Closure_Grid::input_5_closure* genClosure = new Closure_Grid::input_5_closure();
  genClosure->getP0() = c_num;
  genClosure->getP1() = src;
  input(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::input(Closure_Grid::input_5_closure* genClosure){
  if (!__dep.get()) _sdag_init();
  __dep->pushBuffer(0, genClosure, 0);
  SDAG::Continuation* c = __dep->tryFindContinuation(0);
  if (c) {
    _TRACE_END_EXECUTE(); 
    _when_0(
      static_cast<Closure_Grid::printout_3_closure*>(c->closure[0])
    );
    _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
    delete c;
  }
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::_sdag_init() {
  __dep.reset(new SDAG::Dependency(1,1));
  __dep->addDepends(0,0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::__sdag_init() {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Grid::_sdag_pup(PUP::er &p) {
    bool hasSDAG = __dep.get();
    p|hasSDAG;
    if (p.isUnpacking() && hasSDAG) _sdag_init();
    if (hasSDAG) { __dep->pup(p); }
}
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Grid::__sdag_register() {
  (void)_sdag_idx_Grid_atomic_0();
  (void)_sdag_idx_Grid_atomic_1();
  (void)_sdag_idx_Grid_atomic_2();
  (void)_sdag_idx_Grid_atomic_3();
  PUPable_reg(SINGLE_ARG(Closure_Grid::SendInput_2_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::printout_3_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::printmain_4_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::input_5_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::SendInput_2_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::printout_3_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::printmain_4_closure));
  PUPable_reg(SINGLE_ARG(Closure_Grid::input_5_closure));
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_idx_Grid_atomic_0() {
  static int epidx = _sdag_reg_Grid_atomic_0();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_reg_Grid_atomic_0() {
  return CkRegisterEp("Grid_atomic_0", NULL, 0, CkIndex_Grid::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_idx_Grid_atomic_1() {
  static int epidx = _sdag_reg_Grid_atomic_1();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_reg_Grid_atomic_1() {
  return CkRegisterEp("Grid_atomic_1", NULL, 0, CkIndex_Grid::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_idx_Grid_atomic_2() {
  static int epidx = _sdag_reg_Grid_atomic_2();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_reg_Grid_atomic_2() {
  return CkRegisterEp("Grid_atomic_2", NULL, 0, CkIndex_Grid::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_idx_Grid_atomic_3() {
  static int epidx = _sdag_reg_Grid_atomic_3();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Grid::_sdag_reg_Grid_atomic_3() {
  return CkRegisterEp("Grid_atomic_3", NULL, 0, CkIndex_Grid::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */



#ifndef CK_TEMPLATES_ONLY
void _registertest(void)
{
  static int _done = 0; if(_done) return; _done = 1;
  CkRegisterReadonly("mainProxy","CProxy_Main",sizeof(mainProxy),(void *) &mainProxy,__xlater_roPup_mainProxy);

/* REG: mainchare Main: Chare{
Main(CkArgMsg* impl_msg);
void done();
};
*/
  CkIndex_Main::__register("Main", sizeof(Main));

/* REG: array Grid: ArrayElement{
Grid(const bool &accept, int work_num);
void SendInput(const CProxy_Grid &output);
void printout(int num, const CkCallback &cb);
void printmain(int num);
void input(int c_num, const int *src);
void update_res(CkReductionMsg* impl_msg);
Grid(CkMigrateMessage* impl_msg);
};
*/
  CkIndex_Grid::__register("Grid", sizeof(Grid));

}
extern "C" void CkRegisterMainModule(void) {
  _registertest();
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
