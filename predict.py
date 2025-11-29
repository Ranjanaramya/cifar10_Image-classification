# predict.py - Predict single image using saved model
import argparse, os, numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def load_and_prep_image(path):
    img = Image.open(path).convert('RGB').resize((32,32))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def main(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    model = load_model(args.model_path)
    arr = load_and_prep_image(args.image)
    preds = model.predict(arr)
    idx = int(np.argmax(preds[0]))
    print('Predicted:', CLASS_NAMES[idx], 'Confidence:', float(np.max(preds[0])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    main(args)
