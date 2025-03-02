# generate the sentence embeddings of the captions
import glob
import os, json, sys, torch, argparse
import random

import tqdm
from transformers import AutoModel, AutoTokenizer
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentence embeddings of the captions")
    parser.add_argument("--dataset", type=str, choices=['ucf', 'shanghai', 'violence', 'ped2'], default='ucf',
                        help="dataset to generate caption embeddings")
    parser.add_argument("--is_test", action='store_true', default=False,
                        help="whether to generate test caption embeddings")
    parser.add_argument("--caption_path", type=str, default='/media/tcc/新加卷/tcc/tccprogram/datas/ucf_test_caption_v2',
                        help="path to save the generated embeddings")
    parser.add_argument("--output_path", type=str,
                        default='/media/tcc/新加卷/tcc/tccprogram/LAP/save/Crime/llamavideo/',
                        help="path to save the generated embeddings")
    parser.add_argument("--model", type=str, default='sup-simcse-bert-base-uncased',
                        help="name of the pretrained model")
    args = parser.parse_args()
    is_test = "test" if args.is_test else "train"
    if args.dataset == "ucf":
        ds_name = "Crime"
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/Crime/RTFM_train_caption/all_captions.txt"
    elif args.dataset == "shanghai":
        ds_name = "Shanghai"
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/" + ds_name + "/RTFM_train_caption/" + is_test + "_captions.txt"
    elif args.dataset == "violence":
        ds_name = "Violence"
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/Violence/RTFM_train_caption/all_captions.txt"
    elif args.dataset == "ped2":
        ds_name = "UCSDped2"
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/" + ds_name + "/RTFM_train_caption/" + is_test + "_captions.txt"
    else:
        raise ValueError("dataset should be either ucf, shanghai, or violence")

    if args.caption_path != '':
        caption_path = args.caption_path

    print("Loading captions from ", caption_path)
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/" + args.model)
    model = AutoModel.from_pretrained("princeton-nlp/" + args.model)
    # model.to("cuda")
    # tokenizer = None
    # model = None
    file_list = glob.glob("/home/tcc/pxh/ucf_caption/*")
    test_list = []
    normal_test_list = []
    with open("/media/tcc/新加卷/tcc/tccprogram/LAP/list/ucf-clip-test.list", "r") as f:
        origin_test_list = f.readlines()
        for test_item in origin_test_list:
            test_list.append("_".join(test_item.split("/")[-1].split("_")[:-1]))
            if "Nor" in test_item:
                normal_test_list.append(test_item.replace("_", "").split("/")[-1].split("x264")[0])

    for file in tqdm.tqdm(file_list):
        vid_name = file.split("/")[-1].split(".")[0]
        # if "Nor" in vid_name:
        #     # if vid_name.replace("_x264", "").replace("_", "") not in normal_test_list:
        #     #     continue
        #     head, tail = vid_name.split("os_")
        #     vid_name = f"{head}os{tail}"
        emb_path = args.output_path + vid_name + "_emb.npy"
        if os.path.exists(emb_path):
            continue
        # if vid_name not in test_list:
        #     continue
        print(f"process {vid_name}")
        with open(file, "rb") as f:
            video_text_list = []
            origin_text_list = f.readlines()
            for text in origin_text_list:
                text = str(text).replace("\\n", "")
                text = " ".join(text.split(" ")[1:]).strip()
                video_text_list.append(text.replace("'", ""))
            try:
                inputs = tokenizer(video_text_list, padding=True, truncation=True, return_tensors="pt")#.to("cuda")
                with torch.no_grad():
                    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
                np.save(emb_path, embeddings.detach().cpu().numpy())
            except:
                print(f"{vid_name} failed")
    # with open(caption_path) as f:
    #     for line in f:
    #         captions = json.loads(line)
    #         key = []
    #         for k in captions:
    #             key.append(k)
    #         assert len(key) == 1
    #         key = key[0]
    #         if key.endswith("mp4") or key.endswith("avi"):
    #             vid_name = os.path.split(key)[1][:-4]
    #         else:
    #             vid_name = os.path.split(key)[-1]
    #         print(vid_name)
    #         if args.output_path != "":
    #             emb_path = args.output_path + vid_name + "_emb.npy"
    #         else:
    #             emb_path = "/home/acsguser/Codes/SwinBERT/datasets/" + ds_name + "/RTFM_train_caption/sent_emb_n/" + vid_name + "_emb.npy"
    #         texts = captions[key]
    #         inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    #         with torch.no_grad():
    #             embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    #
    #         np.save(emb_path, embeddings)
