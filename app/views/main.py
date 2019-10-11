import os, time, sys
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from flask import render_template, jsonify
from app import app
import random

stripe_keys = {
  'secret_key': "sk_test_GvpPOs0XFxeP0fQiWMmk6HYe",
  'publishable_key': "pk_test_UU62FhsIB6457uPiUX6mJS5x"
}

ALLOWED_EXTENSIONS = set(['wav', 'm4a', '3gp', 'oma', 'mp3', 'mp4'])
models_path = app.config['MODELS_PATH'] 
def allowed_file(filename):
		return '.' in filename and \
					 filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload')
def upload_file2():
	 return render_template('index.html')

@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
	 if request.method == 'POST':
	 		#CLEANUP SCRIPT
			upload_folder = '/opt/render/project/src/app/static' # Debugging, better use 'app.config['UPLOAD_FOLDER']' instead 
			print(upload_folder) #Debugging
			now = time.time()

			for filename in os.listdir(upload_folder):
			    if allowed_file(filename):
				    if os.path.getmtime(os.path.join(upload_folder, filename)) < now - 30 * 60:
				        if os.path.isfile(os.path.join(upload_folder, filename)):
				            print(filename)
				            os.remove(os.path.join(upload_folder, filename))

			f = request.files['file']
			print(f.filename) #Debugging
			if f and allowed_file(f.filename):
				path = upload_folder + '/' + f.filename
				f.save(path)
				# INFERENCE.PY
				import chainer
				from chainer import backends
				import cv2
				import librosa
				import numpy as np
				from tqdm import tqdm

				from lib import spec_utils
				from lib import unet

				#p = argparse.ArgumentParser()
				agpu = -1 #p.add_argument('--gpu', '-g', type=int, default=-1)
				amodel = models_path #p.add_argument('--model', '-m', type=str, default='models/baseline.npz')
				ainput = path #p.add_argument('--input', '-i', required=True)
				asr = 44100 #p.add_argument('--sr', '-r', type=int, default=44100)
				ahop_length = 1024 #p.add_argument('--hop_length', '-l', type=int, default=1024)
				awindow_size = 1024 #p.add_argument('--window_size', '-w', type=int, default=1024)
				aout_mask = False #p.add_argument('--out_mask', '-M', action='store_true')
				#args = p.parse_args()
				print('loading model...', end=' ')
				model = unet.MultiBandUNet()
				chainer.serializers.load_npz(amodel, model)
				if agpu >= 0:
						chainer.backends.cuda.check_cuda_available()
						chainer.backends.cuda.get_device(agpu).use()
						model.to_gpu()
				xp = model.xp
				print('done')

				#CHANGE DURATION FOR PAID VERSION
				print('loading wave source...', end=' ')
				X, sr = librosa.load(
						ainput, asr, False,duration=30.0, dtype=np.float32, res_type='kaiser_fast')
				print('done')

				print('wave source stft...', end=' ')
				X, phase = spec_utils.calc_spec(X, ahop_length, phase=True)
				coeff = X.max()
				X /= coeff
				print('done')

				left = model.offset
				roi_size = awindow_size - left * 2
				right = roi_size + left - (X.shape[2] % left)
				X_pad = np.pad(X, ((0, 0), (0, 0), (left, right)), mode='reflect')

				masks = []
				with chainer.no_backprop_mode(), chainer.using_config('train', False):
						for j in tqdm(range(int(np.ceil(X.shape[2] / roi_size)))):
								start = j * roi_size
								X_window = X_pad[None, :, :, start:start + awindow_size]
								X_tta = np.concatenate([X_window, X_window[:, ::-1, :, :]])

								pred = model(xp.asarray(X_tta))
								pred = backends.cuda.to_cpu(pred.data)
								pred[1] = pred[1, ::-1, :, :]
								masks.append(pred.mean(axis=0))

				mask = np.concatenate(masks, axis=2)[:, :, :X.shape[2]]
				# vocal_pred = X * (1 - mask) * coeff
				# mask = spec_utils.mask_uninformative(mask, vocal_pred)
				inst_pred = X * mask * coeff
				vocal_pred = X * (1 - mask) * coeff

				if aout_mask:
						norm_mask = np.uint8(mask.mean(axis=0) * 255)[::-1]
						hm = cv2.applyColorMap(norm_mask, cv2.COLORMAP_MAGMA)
						cv2.imwrite('mask.png', hm)

				print('instrumental inverse stft...', end=' ')
				wav = spec_utils.spec_to_wav(inst_pred, phase, ahop_length)
				print('done')
				instrumental = f.filename.split('.')[0] + '_instrumental.wav'
				librosa.output.write_wav('app/static/' + instrumental, wav, sr)

				print('vocal inverse stft...', end=' ')
				wav = spec_utils.spec_to_wav(vocal_pred, phase, ahop_length)
				print('done')

				vocal = f.filename.split('.')[0] + '_vocal.wav'				
				librosa.output.write_wav('app/static/' + vocal, wav, sr)
				return render_template('uploaded.html', title='Success', original=f.filename, instrumental=instrumental, vocal=vocal)

@app.route('/')
@app.route('/index')
def index():
		return render_template('index.html', title='Home', key=stripe_keys['publishable_key'])


@app.route('/map/refresh', methods=['POST'])
def map_refresh():
		points = [(random.uniform(48.8434100, 48.8634100),
							 random.uniform(2.3388000, 2.3588000))
							for _ in range(random.randint(2, 9))]
		return jsonify({'points': points})


@app.route('/contact')
def contact():
		return render_template('contact.html', title='Contact')
