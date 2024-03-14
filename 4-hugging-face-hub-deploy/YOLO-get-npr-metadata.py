from ultralytics import YOLO
from matplotlib import pyplot as plt
import torch
import csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def draw_bbox(image_path, model_path, save_path):
    """Detect object, lot bbox, and store"""
    image_filename = os.path.basename(image_path)[:-4]
    model = YOLO(model_path)
    pred_imgs = model.predict(image_path, conf=0.3)
        
    # Iterate over each prediction, plot, and save
    for i, pred_img in enumerate(pred_imgs):
        result_array = pred_img.plot()
        plt.imshow(result_array)
        plt.title(f"pred_{image_filename}") 
        plt.savefig(f'{save_path}\\pred_{image_filename}.jpg', dpi=300)

def get_bbox_info(model_path, image_path):
    """Obtain bounding box attributes"""
    model = YOLO(model_path)
    if image_path.endswith('.jpg'):        
        pred = model(image_path)
        for result in pred:
            boxes = result.boxes
            label= tensor_to_list(boxes.cls)  # class values of the boxes
            conf = tensor_to_list(boxes.conf) # confidence values of the boxes
            xyxy = tensor_to_list(boxes.xyxy) # boxes top left and bottom right
            xywh = tensor_to_list(boxes.xywh) # top left width height
    
        return label, conf, xyxy, xywh

def tensor_to_list(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.tolist()
    else:
        raise TypeError("Input must be a PyTorch tensor")
    
def generate_csv(video_id, frame_id, label, conf, xyxy, xywh, csv_filename):
    mode = 'a' if os.path.exists(csv_filename) else 'w'
    with open(csv_filename, mode, newline='') as csvfile:
        fieldnames = ['file_name',
                      'videoname',
                      'videoID',
                      'frameID',
                      'obj_label',
                      'obj_count',
                      'confi_lvl', 
                      'bbox_xyxy',
                      'bbox_xywh']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header only if the file is newly created
        if mode == 'w':
            writer.writeheader()

        # Define function to map videoID to vid_name
        def map_vid_name(videoID):
            if videoID == 'VID001':
                return "How Green Roofs Can Help Cities"
            elif videoID == 'VID002':
                return "What Does High-Quality Preschool Look Like"
            elif videoID == 'VID003':
                return "Why It's Usually Hotter in a City Let's Talk"
            else:
                return None

        file_name = f"{frame_id}.jpg"

        videoname = map_vid_name(video_id)

        count = len(label)

        writer.writerow({'file_name': file_name,
                         'videoname': videoname,
                         'videoID': video_id,
                         'frameID': frame_id,
                         'obj_label': label,
                         'obj_count': count,
                         'confi_lvl': conf,
                         'bbox_xyxy': xyxy,
                         'bbox_xywh': xywh})
            
def main():
    OUTPUT_FILENAME = 'metadata.csv'
    model_path  = 'YOLO-best.pt'
    root_DATASET = 'D:\\GIT-CS370-IntroductionToAI\\cs370-tn268-introduction-to-ai-assignments\\assignment-2-video-search\\1-youtube-downloader\\DATASET-FRAMES'
    save_path   = 'image-pred-bbox'

    # root_DATASET = 'assignment-2-video-search\\3-embedding-model\\image-pred'
    # save_path = 'assignment-2-video-search\\3-embedding-model\\image-pred-bbox'
    
    video_id_counter = 1
    for folder_name in sorted(os.listdir(root_DATASET)):
        folder_DATASET = os.path.join(root_DATASET, folder_name)
        video_id = f"VID{video_id_counter:03d}" 
        video_id_counter += 1 
        for image in sorted(os.listdir(folder_DATASET)):
            image_path = os.path.join(folder_DATASET, image)
            image_id   = os.path.splitext(image)[0]
            
            # Get bbox info and compile results in csv
            label, conf, xyxy, xywh  = get_bbox_info(model_path, image_path)
            if len(label) == 0:
                continue
            # Use model for object detection, draw bbox and store
            # draw_bbox(image_path, model_path, save_path)
            generate_csv(video_id= video_id,
                         frame_id= image_id,
                         label= label,
                         conf = conf,
                         xyxy = xyxy,
                         xywh = xywh,
                         csv_filename = OUTPUT_FILENAME)
            

if __name__ == "__main__":
    main()
    
