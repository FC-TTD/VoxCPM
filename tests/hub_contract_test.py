"""Native-framework CPU contracts; fake weights, real API/UI/audio/LoRA cache/IPC."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from api.lora_manager import LoRAManager
from hub_runtime.adapter import NativeBackend
from hub_fakes import FakeNative, FakeRuntime


def wav_bytes(sr=24000):
    buffer=BytesIO();sf.write(buffer,np.sin(2*np.pi*220*np.arange(sr)/sr)*.2,sr,format="WAV")
    return buffer.getvalue()


@pytest.fixture
def runtime():
    return FakeRuntime()


@pytest.fixture
def client(runtime):
    from hub_runtime.__main__ import create_app
    with TestClient(create_app(runtime)) as client: yield client


def test_original_openapi_fields_defaults_unchanged():
    from hub_runtime.__main__ import describe
    from api.server import app as native
    expected=native.openapi(); actual=describe()
    def body(schema):
        ref=schema['paths']['/generate']['post']['requestBody']['content']['multipart/form-data']['schema']['$ref']
        value=dict(schema['components']['schemas'][ref.rsplit('/',1)[-1]])
        value.pop('title',None);return value
    assert body(actual)==body(expected)


def test_api_generation_controls_prompt_reference_and_output(client,runtime):
    response=client.post('/generate',data={'text':'hello','control':'excited','cfg_value':'2.4','inference_timesteps':'12',
        'prompt_text':'reference words','normalize':'false','denoise':'true','postprocess':'false','trim_silence':'false'},
        files={'prompt_audio':('p.wav',wav_bytes(),'audio/wav'),'reference_audio':('r.wav',wav_bytes(48000),'audio/wav')})
    assert response.status_code==200,response.text
    wav,sr=sf.read(BytesIO(response.content));assert sr==48000 and wav.size==48000
    call=runtime.backend.model.calls[-1]
    assert call['text']=='(excited)hello' and call['cfg_value']==2.4 and call['inference_timesteps']==12
    assert call['normalize'] is False and call['denoise'] is True
    assert call['prompt_wav_path_sample_rate']==call['reference_wav_path_sample_rate']==16000
    assert not Path(call['prompt_wav_path']).exists() and not Path(call['reference_wav_path']).exists()
    assert runtime.active==0 and runtime.calls==1


@pytest.mark.parametrize('data,files,status',[
    ({'text':'hello','prompt_text':'words'},None,400),
    ({'text':'hello'},{'prompt_audio':('p.wav',wav_bytes(),'audio/wav')},400),
    ({'text':'hello','prompt_text':'words'},{'prompt_audio':('bad.wav',b'bad','audio/wav')},400),
    ({'text':'hello','lora_name':'/does-not-exist'},None,404),
    ({'text':'hello','cfg_value':'bad'},None,422),
])
def test_api_original_errors(client,runtime,data,files,status):
    response=client.post('/generate',data=data,files=files)
    assert response.status_code==status,response.text
    assert runtime.active==0


def test_real_native_audio_postprocess(client):
    response=client.post('/generate',data={'text':'hello','postprocess':'true','trim_silence':'true','lufs':'-21'})
    assert response.status_code==200,response.text
    wav,sr=sf.read(BytesIO(response.content));assert sr==48000 and wav.size>0 and np.isfinite(wav).all()


def make_lora(tmp_path,index):
    path=tmp_path/f'lora-{index}.ckpt';torch.save({'weight':torch.full((2,2),float(index))},path);return str(path)


def test_cpu_lora_lru5_hot_swap_disable_and_ui_state(tmp_path):
    native=FakeNative();cache=LoRAManager(capacity=5);engine=NativeBackend(native,cache)
    paths=[make_lora(tmp_path,i) for i in range(6)]
    for i,path in enumerate(paths):
        engine.generate_managed(path,text='hello')
        assert torch.all(native.tts_model.weight==float(i))
    assert len(cache.cache)==5 and paths[0] not in cache.cache
    assert all(t.device.type=='cpu' for state in cache.cache.values() for t in state.values())
    Path(paths[5]).unlink()
    engine.generate_managed(paths[4],text='hello')
    engine.generate_managed(paths[5],text='hello')
    assert torch.all(native.tts_model.weight==5)
    engine.generate_ui(text='hello');assert native.calls[-1]['active_lora'] is True
    engine.generate_managed(None,text='hello');assert native.calls[-1]['active_lora'] is False
    assert len(cache.cache)==5


def test_failed_lora_reports_original_error_prefix(tmp_path):
    engine=NativeBackend(FakeNative(),LoRAManager())
    with pytest.raises(HTTPException) as failed:engine.generate_managed(str(tmp_path/'missing'),text='hello')
    assert failed.value.status_code==500 and failed.value.detail.startswith('Failed to switch LoRA:')


def test_api_ui_native_calls_are_serialized():
    entered,released=threading.Event(),threading.Event()
    native=FakeNative(gate=(entered,released));engine=NativeBackend(native,LoRAManager())
    with ThreadPoolExecutor(2) as pool:
        first=pool.submit(engine.generate_managed,None,text='api')
        assert entered.wait(5)
        second=pool.submit(engine.generate_ui,text='ui')
        try:
            time.sleep(.05);assert len(native.calls)==1
        finally:released.set()
        first.result(5);second.result(5)
    assert [x['text'] for x in native.calls]==['api','ui']


def test_copied_ui_callback_prompt_semantics_and_no_load_fallback(runtime,tmp_path):
    from hub_runtime.ui import VoxCPMDemo
    demo=VoxCPMDemo(runtime);ref=tmp_path/'ref.wav';ref.write_bytes(wav_bytes())
    sr,wav=demo.generate_tts_audio(' hello ','(excited)（fast）',str(ref),'',2.4,False,True,12)
    assert sr==48000 and wav.size>0
    call=runtime.backend.model.calls[-1]
    assert call['text']=='(excitedfast)hello' and call['reference_wav_path']==str(ref)
    assert 'prompt_wav_path' not in call
    demo.generate_tts_audio('hello','',str(ref),' reference words ')
    call=runtime.backend.model.calls[-1]
    assert call['prompt_wav_path']==call['reference_wav_path']==str(ref)
    assert call['prompt_text']=='reference words'
    runtime.get=lambda: (_ for _ in ()).throw(RuntimeError('admission failed'))
    with pytest.raises(RuntimeError,match='admission failed'):demo.generate_tts_audio('hello')
    assert runtime.active==0


def test_original_gradio6_mount_i18n_ui_defaults_and_health(client,runtime):
    response=client.get('/',follow_redirects=False)
    assert response.status_code==307 and response.headers['location']=='/gradio/'
    assert client.get('/gradio/').status_code==200
    config=client.get('/gradio/config').json()
    assert config['version'].startswith('6.19.')
    assert {x.get('api_name') for x in config['dependencies']}=={'generate','_on_toggle_instant','_run_asr_if_needed'}
    assert config['enable_queue'] is True
    assert client.get('/health').status_code==200 and runtime.started
    sliders=[c for c in config['components'] if c['type']=='slider']
    assert [c['props']['value'] for c in sliders]==[2.0,10]
    # Gradio serves the original translation dictionary, not bare i18n keys.
    translated=client.get('/gradio/?__theme=light',headers={'accept-language':'zh-CN'}).text
    assert 'reference_audio_label' in translated and ('参考音频' in translated or '\u53c2' in translated)
    logo=client.get('/gradio/gradio_api/file=assets/voxcpm_logo.png')
    assert logo.status_code==200 and logo.headers['content-type'].startswith('image/')


def test_health_responsive_while_busy(client,runtime):
    entered,released=threading.Event(),threading.Event();runtime.backend.model.gate=(entered,released)
    with ThreadPoolExecutor(1) as pool:
        future=pool.submit(client.post,'/generate',data={'text':'hello'})
        try:
            assert entered.wait(5)
            before=time.monotonic();assert client.get('/health').status_code==200
            assert time.monotonic()-before<1 and runtime.active==1
        finally:released.set()
        assert future.result(5).status_code==200


def test_describe_and_ui_import_no_torch_or_legacy_plugin():
    result=subprocess.run([sys.executable,'-c',
      "import sys;from hub_runtime.__main__ import describe;describe();from hub_runtime import ui;"
      "assert 'torch' not in sys.modules;assert 'ttd_fastapi_utils' not in sys.modules;"
      "import funasr;assert '/candidate/funasr/' in funasr.__file__"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_real_process_model_rpc_and_exit():
    from ttd_model_runtime.engine import ProcessModel
    from hub_fakes import load_fake_engine
    from hub_runtime.adapter import completion,release,cleanup
    engine=ProcessModel(load_fake_engine,completion,cleanup,release,
         {'gpu':'GPU-00000000-0000-0000-0000-000000000000','generation':1}).start()
    try:
        assert engine.input_sample_rate()==16000 and engine.supports_reference_audio()
        sr,wav=engine.generate_managed(None,text='hello');assert sr==48000 and wav.size==48000
        sr,wav=engine.generate_ui(text='hello');assert sr==48000 and wav.size==48000
    finally:engine.close()
    assert not engine.engine_status()['alive'] and engine.engine_status()['group_empty']


def test_real_gradio_queue_upload_generation_and_file(tmp_path):
    import httpx
    from gradio_client import Client,handle_file
    s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close()
    log=(tmp_path/'server.log').open('w+')
    p=subprocess.Popen([sys.executable,'-m','hub_fakes'],stdout=log,stderr=subprocess.STDOUT,
        env=dict(os.environ,HUB_TEST_PORT=str(port),GRADIO_ANALYTICS_ENABLED='False'))
    url=f'http://127.0.0.1:{port}'
    try:
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            if p.poll() is not None:log.seek(0);pytest.fail(log.read())
            try:
                if httpx.get(url+'/health',timeout=1).status_code==200:break
            except httpx.HTTPError:pass
            time.sleep(.1)
        else:log.seek(0);pytest.fail(log.read())
        client=Client(url+'/gradio/',verbose=False,download_files=False)
        ref=tmp_path/'ref.wav';ref.write_bytes(wav_bytes())
        for audio,prompt_enabled,prompt in [(None,False,''),(handle_file(str(ref)),False,''),(handle_file(str(ref)),True,'hello')]:
            result=client.predict('hello','warm',audio,prompt_enabled,prompt,2,False,False,10,api_name='/generate')
            file_url=result.get('url') if isinstance(result,dict) else result
            response=httpx.get(file_url,timeout=10);assert response.status_code==200,response.text
            wav,sr=sf.read(BytesIO(response.content));assert sr==48000 and wav.size>0
    finally:
        p.terminate()
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        log.close()


def test_native_constructor_and_placement_are_not_rewritten(monkeypatch):
    from voxcpm import VoxCPM
    from hub_runtime.adapter import load_model
    calls=[]
    def loader(**kwargs):
        calls.append(kwargs)
        return FakeNative()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(VoxCPM,'from_pretrained',loader)
    monkeypatch.setenv('VOXCPM_MODEL_DIR','/app/models/openbmb__VoxCPM2')
    engine=load_model()
    kwargs=calls[-1]
    assert kwargs['hf_model_id']=='/app/models/openbmb__VoxCPM2'
    assert kwargs['load_denoiser'] is True and kwargs['optimize'] is False
    assert kwargs['lora_config'].enable_lm and kwargs['lora_config'].enable_dit
    assert not kwargs['lora_config'].enable_proj
    assert 'device' not in kwargs and 'dtype' not in kwargs
    assert engine.lora_manager.capacity==5
