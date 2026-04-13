from clearvoice import ClearVoice
import soundfile as sf
import numpy as np
import librosa
 
""" 
This demo shows that ClearVoice provide batch processing from numpy input to numpy output
"""

##-----Demo One: use MossFormer2_SR_48K model for speech super-resolution -----------------
if True:
    print('testing MossFormer2_SR_48K ...')
    myClearVoice = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])

    audio, sr = sf.read('samples/input_sr.wav')
    ## Input audio must be 48000 Hz
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
    if len(audio.shape) < 2:
    	audio = np.reshape(audio, [1, audio.shape[0]])
    ## simulate batch input
    #audio = np.concatenate((audio, audio), axis=0)
    audio = audio.astype(np.float32)    
    ## audio: [batch, length]
    ##output_wav: [batch, length]
    output_wav = myClearVoice(audio, False)
    sf.write('samples/output_MossFormer2_SR_48K_input_sr.wav', output_wav[0,:], 48000) 

##-----Demo Three: use MossFormer2_SE_48K model for speech enhancement -----------------
if True:
    print(f'testing MossFormer2_SE_48K ...')
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

    audio, sr = sf.read('samples/input.wav')
    ## Input audio must be 48000 Hz
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
    if len(audio.shape) < 2:
    	audio = np.reshape(audio, [1, audio.shape[0]])
    audio = audio.astype(np.float32)
    ## audio: [batch, length]
    ##output_wav: [batch, length]
    output_wav = myClearVoice(audio, False)
    sf.write('samples/output_MossFormer2_SE_48K_batch.wav', output_wav[0,:], 48000)    
      
##-----Demo Four: use FRCRN_SE_16K model for speech enhancement -----------------
if True:
    print('testing FRCRN_SE_16K ...')
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['FRCRN_SE_16K'])

    ##1sd calling method: process an input waveform and return output waveform, then write to output_FRCRN_SE_16K.wav
    audio, sr = sf.read('samples/input.wav')
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio.shape) < 2:
    	audio = np.reshape(audio, [1, audio.shape[0]])
    #audio = np.concatenate((audio, audio), axis=0)
    audio = audio.astype(np.float32)
    ## audio: [batch, length]
    ##output_wav: [batch, length]
    output_wav = myClearVoice(audio, False)
    sf.write('samples/output_FRCRN_SE_16K_batch.wav', output_wav[0, :], 16000)
    
##-----Demo Five: use MossFormerGAN_SE_16K model for speech enhancement -----------------
if True:
    print(f'testing MossFormerGAN_SE_16K ...')
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormerGAN_SE_16K'])

    ##1sd calling method: process the waveform from input.wav and return output waveform, then write to output_MossFormerGAN_SE_16K.wav
    audio, sr = sf.read('samples/input.wav')
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio.shape) < 2:
    	audio = np.reshape(audio, [1, audio.shape[0]])
    #audio = np.concatenate((audio, audio), axis=0)
    audio = audio.astype(np.float32)
    ## audio: [batch, length]
    ##output_wav: [batch, length]
    output_wav = myClearVoice(audio, False)
    sf.write('samples/output_MossFormerGAN_SE_16K_batch.wav', output_wav[0, :], sr)

##-----Demo Six: use MossFormer2_SS_16K model for speech separation -----------------
if True:
    print(f'testing MossFormer2_SS_16K ...')
    myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

    ##1sd calling method: process an input waveform and return output waveform, then write to output_MossFormer2_SS_16K_s1.wav and output_MossFormer2_SS_16K_s2.wav
    num_spks = 2
    audio, sr = sf.read('samples/input_ss.wav')
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio.shape) < 2:
    	audio = np.reshape(audio, [1, audio.shape[0]])
    audio = audio.astype(np.float32)
    ## audio: [batch, length]
    ## output_wav: [spk, batch, length]
    output_wav = myClearVoice(audio, False)
    for spk in range(num_spks):
        output_file = f'samples/output_MossFormer2_SS_16K_batch_spk{spk+1}.wav'
        sf.write(output_file, output_wav[spk, 0, :], 16000)

