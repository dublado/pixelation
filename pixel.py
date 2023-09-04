import sys
from sklearn.cluster import KMeans
from PIL import Image
import moviepy.editor as mp
import numpy as np
import cv2

def pixelate_old(image_path, pixel_size):
    with Image.open(image_path) as img:
        width, height = img.size
        img = img.resize((width // pixel_size, height // pixel_size), resample=Image.NEAREST)
        img = img.resize((width, height), resample=Image.NEAREST)
        return img


def pixelate_video(video_path, pixel_size, num_colors=256):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'pixelated_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    audio = mp.AudioFileClip(video_path)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.quantize(colors=num_colors)
        img = img.resize((width // pixel_size, height // pixel_size), resample=Image.NEAREST)
        img = img.resize((width, height), resample=Image.NEAREST)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    video.release()
    out.release()

    final_video = mp.VideoFileClip(output_path)
    final_video = final_video.set_audio(audio)
    final_video.write_videofile('final_video.mp4', fps=fps, codec='libx264')


def pixelate(image_path, pixel_size, num_colors=256):
    with Image.open(image_path) as img:
        img = img.quantize(colors=num_colors) # reduce the color palette of the image
        width, height = img.size
        img = img.resize((width // pixel_size, height // pixel_size), resample=Image.NEAREST)
        img = img.resize((width, height), resample=Image.NEAREST)
        return img


def ppixelate_video(video_path, pixel_size, num_colors=256):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'pixelated_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    audio = mp.AudioFileClip(video_path)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame).convert('L') # convert to grayscale
        img = img.quantize(colors=num_colors)
        img = img.resize((width // pixel_size, height // pixel_size), resample=Image.NEAREST)
        img = img.resize((width, height), resample=Image.NEAREST)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # convert back to BGR
        out.write(frame)

    video.release()
    out.release()

    final_video = mp.VideoFileClip(output_path)
    final_video = final_video.set_audio(audio)
    final_video.write_videofile('final_video.mp4', fps=fps, codec='libx264')
#pixelated_image = pixelate('image.jpeg', 5, 8)
#pixelated_image.save('pixelated_image.png')


def apixelate_video(video_path, pixel_size, num_colors=256):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'pixelated_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    audio = mp.AudioFileClip(video_path)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((width // pixel_size, height // pixel_size), resample=Image.NEAREST)
        img = img.resize((width, height), resample=Image.NEAREST)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert back to BGR
        out.write(frame)

    video.release()
    out.release()

    final_video = mp.VideoFileClip(output_path)
    final_video = final_video.set_audio(audio)
    final_video.write_videofile('final_video.mp4', fps=fps, codec='libx264')


def xpixelate_video(video_path, pixel_size, num_colors=256):
    def quantize_image(image, num_colors):
        data = np.float32(image.reshape(-1, 3))
        kmeans = KMeans(num_colors)
        labels = kmeans.fit_predict(data)
        palette = kmeans.cluster_centers_.astype(np.uint8)
        return palette[labels].reshape(image.shape)

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'pixelated_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    audio = mp.AudioFileClip(video_path)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((width // pixel_size, height // pixel_size), resample=Image.NEAREST)
        img = img.resize((width, height), resample=Image.NEAREST)
        frame = np.array(img)
        if num_colors < 256:
            frame = quantize_image(frame, num_colors)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert back to BGR
        out.write(frame)

    video.release()
    out.release()

    final_video = mp.VideoFileClip(output_path)
    final_video = final_video.set_audio(audio)
    final_video.write_videofile('final_video.mp4', fps=fps, codec='libx264')

xpixelate_video(sys.argv[1], 5, num_colors=8)

