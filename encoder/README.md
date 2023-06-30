**# Interaction-Function Centric Encoder**



1. Follow [LOVEU-CVPR22-AQTC/README.md](https://github.com/starsholic/LOVEU-CVPR22-AQTC/blob/main/README.md), [EgoVLP/README.md](https://github.com/showlab/EgoVLP/blob/main/README.md), [Afformer/README.md](https://github.com/showlab/afformer/blob/main/README.md) to install required libraries and dataset.



2. Revise the dataset road of those configuration files.



3. Run the script:


```

cd LOVEU-CVPR22-AQTC/encoder

```



For function-para generating and TF-IDF score calculating, first change the data_root in get_tfidf_score.py line 115, then,


```

python get_tfidf_score.py

```



For hand-object-interaction state mining, first prepare the weights of HI-RCNN according to  [Afformer/README.md](https://github.com/showlab/afformer/blob/main/README.md), then, 


```

python main.py --cfg configs/hand.yaml FOR.HAND True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json"

```



**#  Note that paras, videos & hands can be reused in both training and testing sets 2023, so we only need to encode them once ;)**

For video encoding, 


```

python main.py --cfg configs/blip.yaml FOR.VIDEO True  DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json"

```



For script encoding, (Not used)


```

python main.py --cfg configs/blip.yaml FOR.SCRIPT True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json"

```



For function-para encoding,


```

python main.py --cfg configs/blip.yaml FOR.PARA True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json"

```



For QA encoding,


```

python main.py --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "train" DATASET.LABEL "train_with_score.json"

python main.py --cfg configs/vit_xlnet.yaml FOR.QA True DATASET.SPLIT "test2023" DATASET.LABEL "test2023_without_gt_with_score.json"

```





4. Make sure each sample in encoder/outputs has video, script, function_para and QAs embedding:)