import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import glob
import yaml
import numpy as np
from tqdm import tqdm
import soundfile as sf
from model_freq_domain import signal_chain_gpu
from data_processing.data_processing_batch import AudioReader

''''''
CHECKPOINT_PATH = 'checkpoints/12_06_2022_17_14_13_rats_small_train_thd0.8,1.0_sr16000_len88'
CHECKPOINT_STEP = 'ckpt-150'
NOISE_ADJUSTMENT = 4.0

INPUT_DIR = 'data/test'
OUTPUT_DIR = 'data/output'
''''''

def save_audio(pred, sr, audio_name):
	pred =tf.reshape(pred, (1, -1))
	pred = pred.numpy()[0].astype(np.float32)
	sf.write(audio_name, pred, sr,subtype='PCM_16')

if __name__=="__main__":

	with open (CHECKPOINT_PATH+'/training.yaml', 'r') as f:
		cfg = yaml.safe_load(f)
	with open (CHECKPOINT_PATH+'/model.yaml', 'r') as f:
		cfg_model = yaml.safe_load(f)

	os.makedirs(OUTPUT_DIR, exist_ok=True)

	#define model
	model = signal_chain_gpu(
		EQ_cfg = cfg_model['FIRfilter'], 
		DRC_cfg = cfg_model['compressor'],
		waveshaper_cfg = cfg_model['distortion'], 
		noise_cfg = cfg_model['filtered_noise'], 
		noise_adjustment=NOISE_ADJUSTMENT,
		)

	optimizer = tf.keras.optimizers.Adam(
		cfg['learning_rate'], 
		beta_1=0.9, 
		beta_2=0.98,
		epsilon=1e-9)

	ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
	ckpt.restore(os.path.join(CHECKPOINT_PATH, CHECKPOINT_STEP))

	#make data
	clean_paths = [x for x in glob.glob(os.path.join(INPUT_DIR, '*.wav'))]

	@tf.function
	def forward(inp_audio):
		return model(inp_audio)

	for i, clean_path in tqdm(enumerate(clean_paths)):
		print(f' Processing {clean_path}')
		audio = AudioReader(audio_file_path = clean_path, if_pad_end = True, if_noise_floor = True, sample_rate = cfg["sampling_rate"])

		inp_audio = audio.sample_array[np.newaxis, :] #pending change
		out = forward(inp_audio).numpy()

		save_audio(out[:, :audio.len], cfg["sampling_rate"], audio_name=os.path.join(OUTPUT_DIR, os.path.basename(clean_path)))
