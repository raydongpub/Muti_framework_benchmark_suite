
/* ---------------- method closures -------------- */
#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */



/* ---------------- method closures -------------- */
#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Data::pauseForLB_2_closure : public SDAG::Closure {
      

      pauseForLB_2_closure() {
        init();
      }
      pauseForLB_2_closure(CkMigrateMessage*) {
        init();
      }
            void pup(PUP::er& __p) {
        packClosure(__p);
      }
      virtual ~pauseForLB_2_closure() {
      }
      PUPable_decl(SINGLE_ARG(pauseForLB_2_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Data::DataLoad_3_closure : public SDAG::Closure {
            ParticleDataset::Particle *dPar;
            int numPar;

      CkMarshallMsg* _impl_marshall;
      char* _impl_buf_in;
      int _impl_buf_size;

      DataLoad_3_closure() {
        init();
        _impl_marshall = 0;
        _impl_buf_in = 0;
        _impl_buf_size = 0;
      }
      DataLoad_3_closure(CkMigrateMessage*) {
        init();
        _impl_marshall = 0;
        _impl_buf_in = 0;
        _impl_buf_size = 0;
      }
            ParticleDataset::Particle *& getP0() { return dPar;}
            int & getP1() { return numPar;}
      void pup(PUP::er& __p) {
        __p | numPar;
        packClosure(__p);
        __p | _impl_buf_size;
        bool hasMsg = (_impl_marshall != 0); __p | hasMsg;
        if (hasMsg) CkPupMessage(__p, (void**)&_impl_marshall);
        else PUParray(__p, _impl_buf_in, _impl_buf_size);
        if (__p.isUnpacking()) {
          char *impl_buf = _impl_marshall ? _impl_marshall->msgBuf : _impl_buf_in;
          PUP::fromMem implP(impl_buf);
  int impl_off_dPar, impl_cnt_dPar;
  implP|impl_off_dPar;
  implP|impl_cnt_dPar;
  int numPar; implP|numPar;
          impl_buf+=CK_ALIGN(implP.size(),16);
#if !CMK_ONESIDED_IMPL
#endif
          dPar = (ParticleDataset::Particle *)(impl_buf+impl_off_dPar);
        }
      }
      virtual ~DataLoad_3_closure() {
        if (_impl_marshall) CmiFree(UsrToEnv(_impl_marshall));
      }
      PUPable_decl(SINGLE_ARG(DataLoad_3_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Data::CudaCompute_4_closure : public SDAG::Closure {
      

      CudaCompute_4_closure() {
        init();
      }
      CudaCompute_4_closure(CkMigrateMessage*) {
        init();
      }
            void pup(PUP::er& __p) {
        packClosure(__p);
      }
      virtual ~CudaCompute_4_closure() {
      }
      PUPable_decl(SINGLE_ARG(CudaCompute_4_closure));
    };
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY

    struct Closure_Data::IterBegin_5_closure : public SDAG::Closure {
            ParticleDataset::Particle *dPar;
            int numPar;
            CkCallback collect;

      CkMarshallMsg* _impl_marshall;
      char* _impl_buf_in;
      int _impl_buf_size;

      IterBegin_5_closure() {
        init();
        _impl_marshall = 0;
        _impl_buf_in = 0;
        _impl_buf_size = 0;
      }
      IterBegin_5_closure(CkMigrateMessage*) {
        init();
        _impl_marshall = 0;
        _impl_buf_in = 0;
        _impl_buf_size = 0;
      }
            ParticleDataset::Particle *& getP0() { return dPar;}
            int & getP1() { return numPar;}
            CkCallback & getP2() { return collect;}
      void pup(PUP::er& __p) {
        __p | numPar;
        __p | collect;
        packClosure(__p);
        __p | _impl_buf_size;
        bool hasMsg = (_impl_marshall != 0); __p | hasMsg;
        if (hasMsg) CkPupMessage(__p, (void**)&_impl_marshall);
        else PUParray(__p, _impl_buf_in, _impl_buf_size);
        if (__p.isUnpacking()) {
          char *impl_buf = _impl_marshall ? _impl_marshall->msgBuf : _impl_buf_in;
          PUP::fromMem implP(impl_buf);
  int impl_off_dPar, impl_cnt_dPar;
  implP|impl_off_dPar;
  implP|impl_cnt_dPar;
  int numPar; implP|numPar;
  CkCallback collect; implP|collect;
          impl_buf+=CK_ALIGN(implP.size(),16);
#if !CMK_ONESIDED_IMPL
#endif
          dPar = (ParticleDataset::Particle *)(impl_buf+impl_off_dPar);
        }
      }
      virtual ~IterBegin_5_closure() {
        if (_impl_marshall) CmiFree(UsrToEnv(_impl_marshall));
      }
      PUPable_decl(SINGLE_ARG(IterBegin_5_closure));
    };
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
void collect(CkReductionMsg* impl_msg);
void resumeIter(CkReductionMsg* impl_msg);
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
  new (impl_obj_void) Main((CkArgMsg*)impl_msg);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void collect(CkReductionMsg* impl_msg);
 */

void CProxy_Main::collect(CkReductionMsg* impl_msg)
{
  ckCheck();
  if (ckIsDelegated()) {
    int destPE=CkChareMsgPrep(CkIndex_Main::idx_collect_CkReductionMsg(), impl_msg, &ckGetChareID());
    if (destPE!=-1) ckDelegatedTo()->ChareSend(ckDelegatedPtr(),CkIndex_Main::idx_collect_CkReductionMsg(), impl_msg, &ckGetChareID(),destPE);
  } else {
    CkSendMsg(CkIndex_Main::idx_collect_CkReductionMsg(), impl_msg, &ckGetChareID(),0);
  }
}

void CkIndex_Main::_call_redn_wrapper_collect_CkReductionMsg(void* impl_msg, void* impl_obj_void)
{
  Main* impl_obj = static_cast<Main*> (impl_obj_void);
  char* impl_buf = (char*)((CkReductionMsg*)impl_msg)->getData();
  impl_obj->collect((CkReductionMsg*)impl_msg);
  delete (CkReductionMsg*)impl_msg;
}


// Entry point registration function

int CkIndex_Main::reg_collect_CkReductionMsg() {
  int epidx = CkRegisterEp("collect(CkReductionMsg* impl_msg)",
      _call_collect_CkReductionMsg, CMessage_CkReductionMsg::__idx, __idx, 0);
  CkRegisterMessagePupFn(epidx, (CkMessagePupFn)CkReductionMsg::ckDebugPup);
  return epidx;
}


// Redn wrapper registration function

int CkIndex_Main::reg_redn_wrapper_collect_CkReductionMsg() {
  return CkRegisterEp("redn_wrapper_collect(CkReductionMsg *impl_msg)",
      _call_redn_wrapper_collect_CkReductionMsg, CMessage_CkReductionMsg::__idx, __idx, 0);
}


void CkIndex_Main::_call_collect_CkReductionMsg(void* impl_msg, void* impl_obj_void)
{
  Main* impl_obj = static_cast<Main *>(impl_obj_void);
  impl_obj->collect((CkReductionMsg*)impl_msg);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void resumeIter(CkReductionMsg* impl_msg);
 */

void CProxy_Main::resumeIter(CkReductionMsg* impl_msg)
{
  ckCheck();
  if (ckIsDelegated()) {
    int destPE=CkChareMsgPrep(CkIndex_Main::idx_resumeIter_CkReductionMsg(), impl_msg, &ckGetChareID());
    if (destPE!=-1) ckDelegatedTo()->ChareSend(ckDelegatedPtr(),CkIndex_Main::idx_resumeIter_CkReductionMsg(), impl_msg, &ckGetChareID(),destPE);
  } else {
    CkSendMsg(CkIndex_Main::idx_resumeIter_CkReductionMsg(), impl_msg, &ckGetChareID(),0);
  }
}

void CkIndex_Main::_call_redn_wrapper_resumeIter_CkReductionMsg(void* impl_msg, void* impl_obj_void)
{
  Main* impl_obj = static_cast<Main*> (impl_obj_void);
  char* impl_buf = (char*)((CkReductionMsg*)impl_msg)->getData();
  impl_obj->resumeIter((CkReductionMsg*)impl_msg);
  delete (CkReductionMsg*)impl_msg;
}


// Entry point registration function

int CkIndex_Main::reg_resumeIter_CkReductionMsg() {
  int epidx = CkRegisterEp("resumeIter(CkReductionMsg* impl_msg)",
      _call_resumeIter_CkReductionMsg, CMessage_CkReductionMsg::__idx, __idx, 0);
  CkRegisterMessagePupFn(epidx, (CkMessagePupFn)CkReductionMsg::ckDebugPup);
  return epidx;
}


// Redn wrapper registration function

int CkIndex_Main::reg_redn_wrapper_resumeIter_CkReductionMsg() {
  return CkRegisterEp("redn_wrapper_resumeIter(CkReductionMsg *impl_msg)",
      _call_redn_wrapper_resumeIter_CkReductionMsg, CMessage_CkReductionMsg::__idx, __idx, 0);
}


void CkIndex_Main::_call_resumeIter_CkReductionMsg(void* impl_msg, void* impl_obj_void)
{
  Main* impl_obj = static_cast<Main *>(impl_obj_void);
  impl_obj->resumeIter((CkReductionMsg*)impl_msg);
}
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

  // REG: void collect(CkReductionMsg* impl_msg);
  idx_collect_CkReductionMsg();
  idx_redn_wrapper_collect_CkReductionMsg();

  // REG: void resumeIter(CkReductionMsg* impl_msg);
  idx_resumeIter_CkReductionMsg();
  idx_redn_wrapper_resumeIter_CkReductionMsg();

}
#endif /* CK_TEMPLATES_ONLY */


/* DEFS: array Data: ArrayElement{
Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
void pauseForLB();
void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
void CudaCompute();
void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
Data(CkMigrateMessage* impl_msg);
};
 */
#ifndef CK_TEMPLATES_ONLY
 int CkIndex_Data::__idx=0;
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void CProxySection_Data::contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &sid, int userData, int fragSize)
{
   CkArray *ckarr = CProxy_CkArray(sid.get_aid()).ckLocalBranch();
   CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(ckarr->getmCastMgr()).ckLocalBranch();
   mCastGrp->contribute(dataSize, data, type, sid, userData, fragSize);
}

void CProxySection_Data::contribute(int dataSize,void *data,CkReduction::reducerType type, CkSectionInfo &sid, CkCallback &cb, int userData, int fragSize)
{
   CkArray *ckarr = CProxy_CkArray(sid.get_aid()).ckLocalBranch();
   CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(ckarr->getmCastMgr()).ckLocalBranch();
   mCastGrp->contribute(dataSize, data, type, sid, cb, userData, fragSize);
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
/* DEFS: Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
 */

void CProxyElement_Data::insert(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_, int onPE, const CkEntryOptions *impl_e_opts)
{ 
   //Marshall: const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_pL, impl_cnt_pL;
  impl_off_pL=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_pL=sizeof(ParticleDataset::Particle)*(numParticles_));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_pL;
    implP|impl_cnt_pL;
    implP|numParticles_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)grav_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)duration_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)step_;
    implP|penum_;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_pL;
    implP|impl_cnt_pL;
    implP|numParticles_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)grav_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)duration_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)step_;
    implP|penum_;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_pL,pL,impl_cnt_pL);
   UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
   ckInsert((CkArrayMessage *)impl_msg,CkIndex_Data::idx_Data_marshall1(),onPE);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void pauseForLB();
 */

void CProxyElement_Data::pauseForLB(const CkEntryOptions *impl_e_opts) 
{
  static_cast<void>(impl_e_opts);
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Data::idx_pauseForLB_void(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
 */

void CProxyElement_Data::DataLoad(const ParticleDataset::Particle *dPar, int numPar, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const ParticleDataset::Particle *dPar, int numPar
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_dPar, impl_cnt_dPar;
  impl_off_dPar=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_dPar=sizeof(ParticleDataset::Particle)*(numPar));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_dPar,dPar,impl_cnt_dPar);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Data::idx_DataLoad_marshall3(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void CudaCompute();
 */

void CProxyElement_Data::CudaCompute(const CkEntryOptions *impl_e_opts) 
{
  static_cast<void>(impl_e_opts);
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Data::idx_CudaCompute_void(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
 */

void CProxyElement_Data::IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_dPar, impl_cnt_dPar;
  impl_off_dPar=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_dPar=sizeof(ParticleDataset::Particle)*(numPar));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)collect;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)collect;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_dPar,dPar,impl_cnt_dPar);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Data::idx_IterBegin_marshall5(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: Data(CkMigrateMessage* impl_msg);
 */
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
 */

CkArrayID CProxy_Data::ckNew(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_, const CkArrayOptions &opts, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_pL, impl_cnt_pL;
  impl_off_pL=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_pL=sizeof(ParticleDataset::Particle)*(numParticles_));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_pL;
    implP|impl_cnt_pL;
    implP|numParticles_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)grav_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)duration_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)step_;
    implP|penum_;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_pL;
    implP|impl_cnt_pL;
    implP|numParticles_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)grav_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)duration_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)step_;
    implP|penum_;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_pL,pL,impl_cnt_pL);
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkArrayID gId = ckCreateArray((CkArrayMessage *)impl_msg, CkIndex_Data::idx_Data_marshall1(), opts);
  return gId;
}

void CProxy_Data::ckNew(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_, const CkArrayOptions &opts, CkCallback _ck_array_creation_cb, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_pL, impl_cnt_pL;
  impl_off_pL=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_pL=sizeof(ParticleDataset::Particle)*(numParticles_));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_pL;
    implP|impl_cnt_pL;
    implP|numParticles_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)grav_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)duration_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)step_;
    implP|penum_;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_pL;
    implP|impl_cnt_pL;
    implP|numParticles_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)grav_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)duration_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)step_;
    implP|penum_;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_pL,pL,impl_cnt_pL);
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkSendAsyncCreateArray(CkIndex_Data::idx_Data_marshall1(), _ck_array_creation_cb, opts, impl_msg);
}

CkArrayID CProxy_Data::ckNew(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_, const int s1, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_pL, impl_cnt_pL;
  impl_off_pL=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_pL=sizeof(ParticleDataset::Particle)*(numParticles_));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_pL;
    implP|impl_cnt_pL;
    implP|numParticles_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)grav_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)duration_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)step_;
    implP|penum_;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_pL;
    implP|impl_cnt_pL;
    implP|numParticles_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)grav_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)duration_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)step_;
    implP|penum_;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_pL,pL,impl_cnt_pL);
  CkArrayOptions opts(s1);
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkArrayID gId = ckCreateArray((CkArrayMessage *)impl_msg, CkIndex_Data::idx_Data_marshall1(), opts);
  return gId;
}

void CProxy_Data::ckNew(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_, const int s1, CkCallback _ck_array_creation_cb, const CkEntryOptions *impl_e_opts)
{
  //Marshall: const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_pL, impl_cnt_pL;
  impl_off_pL=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_pL=sizeof(ParticleDataset::Particle)*(numParticles_));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_pL;
    implP|impl_cnt_pL;
    implP|numParticles_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)grav_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)duration_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)step_;
    implP|penum_;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_pL;
    implP|impl_cnt_pL;
    implP|numParticles_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)grav_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)duration_;
    //Have to cast away const-ness to get pup routine
    implP|(PRECISION &)step_;
    implP|penum_;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_pL,pL,impl_cnt_pL);
  CkArrayOptions opts(s1);
  UsrToEnv(impl_msg)->setMsgtype(ArrayEltInitMsg);
  CkSendAsyncCreateArray(CkIndex_Data::idx_Data_marshall1(), _ck_array_creation_cb, opts, impl_msg);
}

// Entry point registration function

int CkIndex_Data::reg_Data_marshall1() {
  int epidx = CkRegisterEp("Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_)",
      _call_Data_marshall1, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_Data_marshall1);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_Data_marshall1);

  return epidx;
}


void CkIndex_Data::_call_Data_marshall1(void* impl_msg, void* impl_obj_void)
{
  Data* impl_obj = static_cast<Data *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_*/
  PUP::fromMem implP(impl_buf);
  int impl_off_pL, impl_cnt_pL;
  implP|impl_off_pL;
  implP|impl_cnt_pL;
  int numParticles_; implP|numParticles_;
  PRECISION grav_; implP|grav_;
  PRECISION duration_; implP|duration_;
  PRECISION step_; implP|step_;
  int penum_; implP|penum_;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  ParticleDataset::Particle *pL=(ParticleDataset::Particle *)(impl_buf+impl_off_pL);
  new (impl_obj_void) Data(pL, numParticles_, grav_, duration_, step_, penum_);
}

int CkIndex_Data::_callmarshall_Data_marshall1(char* impl_buf, void* impl_obj_void) {
  Data* impl_obj = static_cast< Data *>(impl_obj_void);
  /*Unmarshall pup'd fields: const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_*/
  PUP::fromMem implP(impl_buf);
  int impl_off_pL, impl_cnt_pL;
  implP|impl_off_pL;
  implP|impl_cnt_pL;
  int numParticles_; implP|numParticles_;
  PRECISION grav_; implP|grav_;
  PRECISION duration_; implP|duration_;
  PRECISION step_; implP|step_;
  int penum_; implP|penum_;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  ParticleDataset::Particle *pL=(ParticleDataset::Particle *)(impl_buf+impl_off_pL);
  new (impl_obj_void) Data(pL, numParticles_, grav_, duration_, step_, penum_);
  return implP.size();
}

void CkIndex_Data::_marshallmessagepup_Data_marshall1(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_*/
  PUP::fromMem implP(impl_buf);
  int impl_off_pL, impl_cnt_pL;
  implP|impl_off_pL;
  implP|impl_cnt_pL;
  int numParticles_; implP|numParticles_;
  PRECISION grav_; implP|grav_;
  PRECISION duration_; implP|duration_;
  PRECISION step_; implP|step_;
  int penum_; implP|penum_;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  ParticleDataset::Particle *pL=(ParticleDataset::Particle *)(impl_buf+impl_off_pL);
  if (implDestP.hasComments()) implDestP.comment("pL");
  implDestP.synchronize(PUP::sync_begin_array);
  for (int impl_i=0;impl_i*(sizeof(*pL))<impl_cnt_pL;impl_i++) {
    implDestP.synchronize(PUP::sync_item);
    implDestP|pL[impl_i];
  }
  implDestP.synchronize(PUP::sync_end_array);
  if (implDestP.hasComments()) implDestP.comment("numParticles_");
  implDestP|numParticles_;
  if (implDestP.hasComments()) implDestP.comment("grav_");
  implDestP|grav_;
  if (implDestP.hasComments()) implDestP.comment("duration_");
  implDestP|duration_;
  if (implDestP.hasComments()) implDestP.comment("step_");
  implDestP|step_;
  if (implDestP.hasComments()) implDestP.comment("penum_");
  implDestP|penum_;
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void pauseForLB();
 */

void CProxy_Data::pauseForLB(const CkEntryOptions *impl_e_opts) 
{
  static_cast<void>(impl_e_opts);
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Data::idx_pauseForLB_void(),0);
}

// Entry point registration function

int CkIndex_Data::reg_pauseForLB_void() {
  int epidx = CkRegisterEp("pauseForLB()",
      _call_pauseForLB_void, 0, __idx, 0);
  return epidx;
}


void CkIndex_Data::_call_pauseForLB_void(void* impl_msg, void* impl_obj_void)
{
  Data* impl_obj = static_cast<Data *>(impl_obj_void);
  CkFreeSysMsg(impl_msg);
  impl_obj->pauseForLB();
}
PUPable_def(SINGLE_ARG(Closure_Data::pauseForLB_2_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
 */

void CProxy_Data::DataLoad(const ParticleDataset::Particle *dPar, int numPar, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const ParticleDataset::Particle *dPar, int numPar
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_dPar, impl_cnt_dPar;
  impl_off_dPar=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_dPar=sizeof(ParticleDataset::Particle)*(numPar));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_dPar,dPar,impl_cnt_dPar);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Data::idx_DataLoad_marshall3(),0);
}

// Entry point registration function

int CkIndex_Data::reg_DataLoad_marshall3() {
  int epidx = CkRegisterEp("DataLoad(const ParticleDataset::Particle *dPar, int numPar)",
      _call_DataLoad_marshall3, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_DataLoad_marshall3);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_DataLoad_marshall3);

  return epidx;
}


void CkIndex_Data::_call_DataLoad_marshall3(void* impl_msg, void* impl_obj_void)
{
  Data* impl_obj = static_cast<Data *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const ParticleDataset::Particle *dPar, int numPar*/
  PUP::fromMem implP(impl_buf);
  int impl_off_dPar, impl_cnt_dPar;
  implP|impl_off_dPar;
  implP|impl_cnt_dPar;
  int numPar; implP|numPar;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  ParticleDataset::Particle *dPar=(ParticleDataset::Particle *)(impl_buf+impl_off_dPar);
  impl_obj->DataLoad(dPar, numPar);
}

int CkIndex_Data::_callmarshall_DataLoad_marshall3(char* impl_buf, void* impl_obj_void) {
  Data* impl_obj = static_cast< Data *>(impl_obj_void);
  /*Unmarshall pup'd fields: const ParticleDataset::Particle *dPar, int numPar*/
  PUP::fromMem implP(impl_buf);
  int impl_off_dPar, impl_cnt_dPar;
  implP|impl_off_dPar;
  implP|impl_cnt_dPar;
  int numPar; implP|numPar;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  ParticleDataset::Particle *dPar=(ParticleDataset::Particle *)(impl_buf+impl_off_dPar);
  impl_obj->DataLoad(dPar, numPar);
  return implP.size();
}

void CkIndex_Data::_marshallmessagepup_DataLoad_marshall3(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const ParticleDataset::Particle *dPar, int numPar*/
  PUP::fromMem implP(impl_buf);
  int impl_off_dPar, impl_cnt_dPar;
  implP|impl_off_dPar;
  implP|impl_cnt_dPar;
  int numPar; implP|numPar;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  ParticleDataset::Particle *dPar=(ParticleDataset::Particle *)(impl_buf+impl_off_dPar);
  if (implDestP.hasComments()) implDestP.comment("dPar");
  implDestP.synchronize(PUP::sync_begin_array);
  for (int impl_i=0;impl_i*(sizeof(*dPar))<impl_cnt_dPar;impl_i++) {
    implDestP.synchronize(PUP::sync_item);
    implDestP|dPar[impl_i];
  }
  implDestP.synchronize(PUP::sync_end_array);
  if (implDestP.hasComments()) implDestP.comment("numPar");
  implDestP|numPar;
}
PUPable_def(SINGLE_ARG(Closure_Data::DataLoad_3_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void CudaCompute();
 */

void CProxy_Data::CudaCompute(const CkEntryOptions *impl_e_opts) 
{
  static_cast<void>(impl_e_opts);
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Data::idx_CudaCompute_void(),0);
}

// Entry point registration function

int CkIndex_Data::reg_CudaCompute_void() {
  int epidx = CkRegisterEp("CudaCompute()",
      _call_CudaCompute_void, 0, __idx, 0);
  return epidx;
}


void CkIndex_Data::_call_CudaCompute_void(void* impl_msg, void* impl_obj_void)
{
  Data* impl_obj = static_cast<Data *>(impl_obj_void);
  CkFreeSysMsg(impl_msg);
  impl_obj->CudaCompute();
}
PUPable_def(SINGLE_ARG(Closure_Data::CudaCompute_4_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
 */

void CProxy_Data::IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_dPar, impl_cnt_dPar;
  impl_off_dPar=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_dPar=sizeof(ParticleDataset::Particle)*(numPar));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)collect;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)collect;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_dPar,dPar,impl_cnt_dPar);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckBroadcast(impl_amsg, CkIndex_Data::idx_IterBegin_marshall5(),0);
}

// Entry point registration function

int CkIndex_Data::reg_IterBegin_marshall5() {
  int epidx = CkRegisterEp("IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect)",
      _call_IterBegin_marshall5, CkMarshallMsg::__idx, __idx, 0+CK_EP_NOKEEP);
  CkRegisterMarshallUnpackFn(epidx, _callmarshall_IterBegin_marshall5);
  CkRegisterMessagePupFn(epidx, _marshallmessagepup_IterBegin_marshall5);

  return epidx;
}


void CkIndex_Data::_call_IterBegin_marshall5(void* impl_msg, void* impl_obj_void)
{
  Data* impl_obj = static_cast<Data *>(impl_obj_void);
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  PUP::fromMem implP(impl_buf);
  Closure_Data::IterBegin_5_closure* genClosure = new Closure_Data::IterBegin_5_closure();
  int impl_off_dPar, impl_cnt_dPar;
  implP|impl_off_dPar;
  implP|impl_cnt_dPar;
  implP|genClosure->numPar;
  implP|genClosure->collect;
  impl_buf+=CK_ALIGN(implP.size(),16);
  genClosure->dPar = (ParticleDataset::Particle *)(impl_buf+impl_off_dPar);
  genClosure->_impl_marshall = impl_msg_typed;
  CmiReference(UsrToEnv(genClosure->_impl_marshall));
  impl_obj->IterBegin(genClosure);
  genClosure->deref();
}

int CkIndex_Data::_callmarshall_IterBegin_marshall5(char* impl_buf, void* impl_obj_void) {
  Data* impl_obj = static_cast< Data *>(impl_obj_void);
  PUP::fromMem implP(impl_buf);
  Closure_Data::IterBegin_5_closure* genClosure = new Closure_Data::IterBegin_5_closure();
  int impl_off_dPar, impl_cnt_dPar;
  implP|impl_off_dPar;
  implP|impl_cnt_dPar;
  implP|genClosure->numPar;
  implP|genClosure->collect;
  impl_buf+=CK_ALIGN(implP.size(),16);
  genClosure->dPar = (ParticleDataset::Particle *)(impl_buf+impl_off_dPar);
  genClosure->_impl_buf_in = impl_buf;
  genClosure->_impl_buf_size = implP.size();
  impl_obj->IterBegin(genClosure);
  genClosure->deref();
  return implP.size();
}

void CkIndex_Data::_marshallmessagepup_IterBegin_marshall5(PUP::er &implDestP,void *impl_msg) {
  CkMarshallMsg *impl_msg_typed=(CkMarshallMsg *)impl_msg;
  char *impl_buf=impl_msg_typed->msgBuf;
  /*Unmarshall pup'd fields: const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect*/
  PUP::fromMem implP(impl_buf);
  int impl_off_dPar, impl_cnt_dPar;
  implP|impl_off_dPar;
  implP|impl_cnt_dPar;
  int numPar; implP|numPar;
  CkCallback collect; implP|collect;
  impl_buf+=CK_ALIGN(implP.size(),16);
  /*Unmarshall arrays:*/
  ParticleDataset::Particle *dPar=(ParticleDataset::Particle *)(impl_buf+impl_off_dPar);
  if (implDestP.hasComments()) implDestP.comment("dPar");
  implDestP.synchronize(PUP::sync_begin_array);
  for (int impl_i=0;impl_i*(sizeof(*dPar))<impl_cnt_dPar;impl_i++) {
    implDestP.synchronize(PUP::sync_item);
    implDestP|dPar[impl_i];
  }
  implDestP.synchronize(PUP::sync_end_array);
  if (implDestP.hasComments()) implDestP.comment("numPar");
  implDestP|numPar;
  if (implDestP.hasComments()) implDestP.comment("collect");
  implDestP|collect;
}
PUPable_def(SINGLE_ARG(Closure_Data::IterBegin_5_closure))
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: Data(CkMigrateMessage* impl_msg);
 */

// Entry point registration function

int CkIndex_Data::reg_Data_CkMigrateMessage() {
  int epidx = CkRegisterEp("Data(CkMigrateMessage* impl_msg)",
      _call_Data_CkMigrateMessage, 0, __idx, 0);
  return epidx;
}


void CkIndex_Data::_call_Data_CkMigrateMessage(void* impl_msg, void* impl_obj_void)
{
  call_migration_constructor<Data> c = impl_obj_void;
  c((CkMigrateMessage*)impl_msg);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
 */
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void pauseForLB();
 */

void CProxySection_Data::pauseForLB(const CkEntryOptions *impl_e_opts) 
{
  static_cast<void>(impl_e_opts);
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Data::idx_pauseForLB_void(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
 */

void CProxySection_Data::DataLoad(const ParticleDataset::Particle *dPar, int numPar, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const ParticleDataset::Particle *dPar, int numPar
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_dPar, impl_cnt_dPar;
  impl_off_dPar=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_dPar=sizeof(ParticleDataset::Particle)*(numPar));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_dPar,dPar,impl_cnt_dPar);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Data::idx_DataLoad_marshall3(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void CudaCompute();
 */

void CProxySection_Data::CudaCompute(const CkEntryOptions *impl_e_opts) 
{
  static_cast<void>(impl_e_opts);
  ckCheck();
  void *impl_msg = CkAllocSysMsg();
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Data::idx_CudaCompute_void(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
 */

void CProxySection_Data::IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect, const CkEntryOptions *impl_e_opts) 
{
  ckCheck();
  //Marshall: const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect
  int impl_off=0;
  int impl_arrstart=0;
  int impl_off_dPar, impl_cnt_dPar;
  impl_off_dPar=impl_off=CK_ALIGN(impl_off,sizeof(ParticleDataset::Particle));
  impl_off+=(impl_cnt_dPar=sizeof(ParticleDataset::Particle)*(numPar));
  { //Find the size of the PUP'd data
    PUP::sizer implP;
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)collect;
    impl_arrstart=CK_ALIGN(implP.size(),16);
    impl_off+=impl_arrstart;
  }
  CkMarshallMsg *impl_msg=CkAllocateMarshallMsg(impl_off,impl_e_opts);
  { //Copy over the PUP'd data
    PUP::toMem implP((void *)impl_msg->msgBuf);
    implP|impl_off_dPar;
    implP|impl_cnt_dPar;
    implP|numPar;
    //Have to cast away const-ness to get pup routine
    implP|(CkCallback &)collect;
  }
  char *impl_buf=impl_msg->msgBuf+impl_arrstart;
  memcpy(impl_buf+impl_off_dPar,dPar,impl_cnt_dPar);
  UsrToEnv(impl_msg)->setMsgtype(ForArrayEltMsg);
  CkArrayMessage *impl_amsg=(CkArrayMessage *)impl_msg;
  impl_amsg->array_setIfNotThere(CkArray_IfNotThere_buffer);
  ckSend(impl_amsg, CkIndex_Data::idx_IterBegin_marshall5(),0);
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
/* DEFS: Data(CkMigrateMessage* impl_msg);
 */
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void CkIndex_Data::__register(const char *s, size_t size) {
  __idx = CkRegisterChare(s, size, TypeArray);
  CkRegisterBase(__idx, CkIndex_ArrayElement::__idx);
  // REG: Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
  idx_Data_marshall1();

  // REG: void pauseForLB();
  idx_pauseForLB_void();

  // REG: void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
  idx_DataLoad_marshall3();

  // REG: void CudaCompute();
  idx_CudaCompute_void();

  // REG: void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
  idx_IterBegin_marshall5();

  // REG: Data(CkMigrateMessage* impl_msg);
  idx_Data_CkMigrateMessage();
  CkRegisterMigCtor(__idx, idx_Data_CkMigrateMessage());

  Data::__sdag_register(); // Potentially missing Data_SDAG_CODE in your class definition?
}
#endif /* CK_TEMPLATES_ONLY */

#ifndef CK_TEMPLATES_ONLY
void Data::IterBegin(ParticleDataset::Particle * dPar, int numPar, CkCallback collect){
  Closure_Data::IterBegin_5_closure* genClosure = new Closure_Data::IterBegin_5_closure();
  genClosure->getP0() = dPar;
  genClosure->getP1() = numPar;
  genClosure->getP2() = collect;
  IterBegin(genClosure);
  genClosure->deref();
}

#endif /* CK_TEMPLATES_ONLY */
#ifndef CK_TEMPLATES_ONLY
void Data::IterBegin(Closure_Data::IterBegin_5_closure* gen0) {
  _TRACE_END_EXECUTE(); 
  if (!__dep.get()) _sdag_init();
  _slist_0(gen0);
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _sdagEP, CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Data::IterBegin_end(Closure_Data::IterBegin_5_closure* gen0) {
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Data::_slist_0(Closure_Data::IterBegin_5_closure* gen0) {
  _serial_0(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Data::_slist_0_end(Closure_Data::IterBegin_5_closure* gen0) {
  IterBegin_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Data::_serial_0(Closure_Data::IterBegin_5_closure* gen0) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Data_serial_0()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    ParticleDataset::Particle*& dPar = gen0->getP0();
    int& numPar = gen0->getP1();
    CkCallback& collect = gen0->getP2();
    { // begin serial block
#line 15 "nbody.ci"

                DataLoad(dPar, numPar);
                CudaCompute();
            
#line 1321 "nbody.def.h"
    } // end serial block
  }
  _TRACE_END_EXECUTE(); 
  _serial_1(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Data::_serial_1(Closure_Data::IterBegin_5_closure* gen0) {
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, (_sdag_idx_Data_serial_1()), CkMyPe(), 0, ckGetArrayIndex().getProjectionID(), this); 
  {
    ParticleDataset::Particle*& dPar = gen0->getP0();
    int& numPar = gen0->getP1();
    CkCallback& collect = gen0->getP2();
    { // begin serial block
#line 19 "nbody.ci"

                CkPrintf("\t\tData: %f\n", localBuf[1].xPos);
                contribute(localCnt*sizeof(ParticleDataset::Particle), localBuf, CkReduction::set, collect);
            
#line 1343 "nbody.def.h"
    } // end serial block
  }
  _TRACE_END_EXECUTE(); 
  _slist_0_end(gen0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Data::_sdag_init() { // Potentially missing Data_SDAG_CODE in your class definition?
  __dep.reset(new SDAG::Dependency(0,0));
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Data::__sdag_init() { // Potentially missing Data_SDAG_CODE in your class definition?
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
void Data::_sdag_pup(PUP::er &p) { // Potentially missing Data_SDAG_CODE in your class definition?
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
void Data::__sdag_register() { // Potentially missing Data_SDAG_CODE in your class definition?
  (void)_sdag_idx_Data_serial_0();
  (void)_sdag_idx_Data_serial_1();
  PUPable_reg(SINGLE_ARG(Closure_Data::pauseForLB_2_closure));
  PUPable_reg(SINGLE_ARG(Closure_Data::DataLoad_3_closure));
  PUPable_reg(SINGLE_ARG(Closure_Data::CudaCompute_4_closure));
  PUPable_reg(SINGLE_ARG(Closure_Data::IterBegin_5_closure));
  PUPable_reg(SINGLE_ARG(Closure_Data::pauseForLB_2_closure));
  PUPable_reg(SINGLE_ARG(Closure_Data::DataLoad_3_closure));
  PUPable_reg(SINGLE_ARG(Closure_Data::CudaCompute_4_closure));
  PUPable_reg(SINGLE_ARG(Closure_Data::IterBegin_5_closure));
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Data::_sdag_idx_Data_serial_0() { // Potentially missing Data_SDAG_CODE in your class definition?
  static int epidx = _sdag_reg_Data_serial_0();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Data::_sdag_reg_Data_serial_0() { // Potentially missing Data_SDAG_CODE in your class definition?
  return CkRegisterEp("Data_serial_0", NULL, 0, CkIndex_Data::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Data::_sdag_idx_Data_serial_1() { // Potentially missing Data_SDAG_CODE in your class definition?
  static int epidx = _sdag_reg_Data_serial_1();
  return epidx;
}
#endif /* CK_TEMPLATES_ONLY */


#ifndef CK_TEMPLATES_ONLY
int Data::_sdag_reg_Data_serial_1() { // Potentially missing Data_SDAG_CODE in your class definition?
  return CkRegisterEp("Data_serial_1", NULL, 0, CkIndex_Data::__idx, 0);
}
#endif /* CK_TEMPLATES_ONLY */



#ifndef CK_TEMPLATES_ONLY
void _registernbody(void)
{
  static int _done = 0; if(_done) return; _done = 1;
  CkRegisterReadonly("mainProxy","CProxy_Main",sizeof(mainProxy),(void *) &mainProxy,__xlater_roPup_mainProxy);

/* REG: mainchare Main: Chare{
Main(CkArgMsg* impl_msg);
void collect(CkReductionMsg* impl_msg);
void resumeIter(CkReductionMsg* impl_msg);
};
*/
  CkIndex_Main::__register("Main", sizeof(Main));


/* REG: array Data: ArrayElement{
Data(const ParticleDataset::Particle *pL, int numParticles_, const PRECISION &grav_, const PRECISION &duration_, const PRECISION &step_, int penum_);
void pauseForLB();
void DataLoad(const ParticleDataset::Particle *dPar, int numPar);
void CudaCompute();
void IterBegin(const ParticleDataset::Particle *dPar, int numPar, const CkCallback &collect);
Data(CkMigrateMessage* impl_msg);
};
*/
  CkIndex_Data::__register("Data", sizeof(Data));

}
extern "C" void CkRegisterMainModule(void) {
  _registernbody();
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
void CBase_Data::virtual_pup(PUP::er &p) {
    recursive_pup<Data >(dynamic_cast<Data* >(this), p);
}
#endif /* CK_TEMPLATES_ONLY */
