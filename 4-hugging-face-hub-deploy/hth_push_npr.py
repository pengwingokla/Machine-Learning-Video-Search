from datasets import load_dataset

dataset = load_dataset("imagefolder", 
                       data_dir='D:\\HFH-Dataset-YouTube-Video-Search\\npr-frames')

dataset.push_to_hub("chloecodes/npr-frames")