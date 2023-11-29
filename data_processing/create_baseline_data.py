import glob 
import os
import subprocess
import soundfile as sf
import numpy as np
# training_datas = glob.glob("/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_clean_8k/raw/org/data_src_train_8k/data/**/*.wav")
training_datas = glob.glob("/home/emrys/G/zixun/espnet/egs2/asr_rl/clean/*.wav")

method = "g726"
# method = "add_rats_noise"
# method = "codec2"
rat_noise, _ = sf.read("/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/rats_noise.wav")
sr = 8000

def concat_audio(input_arr: np.ndarray, noise_arr: np.ndarray, max_length: int) -> np.ndarray:

	if len(input_arr) > max_length:
		output_arr = input_arr[:max_length]
	elif input_arr < max_length:
		output_arr = np.pad(input_arr, (0, max_length-len(input_arr)), mode="minimum")
	else:
		output_arr = input_arr
	
	output_arr = output_arr + noise_arr[:max_length]
	return output_arr


def simulate_g726(input_path: str, output_path: str, arr_noise: np.ndarray, target_sr: int = 8000) -> np.ndarray:
	# ffmpeg will error out if input and output paths are the same
	temp1_path, temp2_path, temp3_path = "temp1.wav", "temp2.wav", "temp3.wav"

	commands = [
		f"ffmpeg -hide_banner -loglevel error -i {input_path} -ar {target_sr} -ac 1 -y {temp1_path}",
		f"ffmpeg -hide_banner -loglevel error -i {temp1_path} -acodec g726 -b:a 16000 {temp2_path}",
		f"ffmpeg -hide_banner -loglevel error -i {temp2_path} -ar {target_sr} -y {temp3_path}",
	]

	for command in commands:
		subprocess.run(command.split(" "))

	arr, _ = sf.read(input_path)
	# ensure array is mono-channelled
	if np.dim(arr) > 1:
		arr = np.mean(arr, axis=1)

	arr_g726, sr = sf.read(temp3_path)
	assert sr == target_sr, f"input sr ({sr}) and target sr ({target_sr}) mismatch."

	out = concat_audio(arr_g726, arr_noise, len(arr))
	sf.write(output_path, out, target_sr)
	subprocess.run(["rm", temp1_path, temp2_path, temp3_path])
	
	return out

def simulate_codec2(input_path: str, output_path: str, arr_noise: np.ndarray, target_sr: int = 8000) -> np.ndarray:

	# ffmpeg will error out if input and output paths are the same
	temp1_path, temp2_path, temp3_path, temp4_path = "temp1.raw", "temp2.bit", "temp3.raw", "temp4.wav"
	br = "700c"

	commands = [
		f"ffmpeg -hide_banner -loglevel error -i {input_path} -f s16le -ar {target_sr} -ac 1 -acodec pcm_s16le {temp1_path}",
		f"c2enc {br} {temp1_path} {temp2_path}",
		f"c2dec {br} {temp2_path} {temp3_path}",
		f"ffmpeg -hide_banner -loglevel error -f s16le -ar {target_sr} -ac 1 -i {temp3_path} {temp4_path}",		
	]

	for command in commands:
		subprocess.run(command.split(" "))

	arr, _ = sf.read(input_path)
	# ensure array is mono-channelled
	if np.dim(arr) > 1:
		arr = np.mean(arr, axis=1)

	arr_codec2, sr = sf.read(temp4_path)
	assert sr == target_sr, f"input sr ({sr}) and target sr ({target_sr}) mismatch."

	out = concat_audio(arr_codec2, arr_noise, len(arr))
	sf.write(output_path, out, target_sr)
	subprocess.run(["rm", temp1_path, temp2_path, temp3_path, temp4_path])

	return out

def augment(audio_path, method):
	if method == "g726":
		return simulate_g726(audio_path, audio_path.replace(".wav", "_out.wav"), rat_noise, 8000)
	elif method == "codec2":
		return simulate_codec2(audio_path, audio_path.replace(".wav", "_out.wav"), rat_noise, 8000)
	else:
		audio_array, _ = sf.read(audio_path)
		return audio_array+rat_noise[:len(audio_array)]

# for training_data in training_datas:
# 	new_dir = f"/home/emrys/G/zixun/espnet/egs2/asr_rl/dump_raw/raw/org/data_src_train_8k_{method}/data/"+training_data.split("/")[-2] 
# 	if not os.path.exists(new_dir):
# 		os.makedirs(new_dir)
# 	new_file_name = new_dir+"/"+training_data.split("/")[-1] 
# 	print("processing input data", training_data)
# 	train_audio_aug = augment(training_data, method = method, outname = new_file_name)

# 	print("new dir", new_dir, new_file_name)
# 	print("saving", new_file_name)

# 	sf.write(new_file_name, train_audio_aug, sr,subtype='PCM_16')

for training_data in training_datas:
	new_dir = f"/home/emrys/G/zixun/espnet/egs2/asr_rl/clean_{method}"
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)
	new_file_name = new_dir+"/"+training_data.split("/")[-1] 
	print("processing input data", training_data)
	train_audio_aug = augment(training_data, method = method, outname = new_file_name)

	print("new dir", new_dir, new_file_name)
	print("saving", new_file_name)

	sf.write(new_file_name, train_audio_aug, sr,subtype='PCM_16')



# out = augment("/home/emrys/G/zixun/espnet/egs2/asr_rl/10280_eng_src_0-13.8140-18.620.wav", method)
# sf.write(f"paper_{method}.wav", out, sr,subtype='PCM_16')

# sf.read("out.wav")

