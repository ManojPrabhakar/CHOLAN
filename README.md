# CHOLAN - [Q105079136](https://www.wikidata.org/wiki/Q105079136) #

CHOLAN : A Modular Approach for Neural Entity Linking on Wikipedia and Wikidata ([paper](https://aclanthology.org/2021.eacl-main.40/))

## Wikidata

* Dataset - We have extracted an EL dataset from the ([T-Rex dataset](https://hadyelsahar.github.io/t-rex/)). Please refer this ([link](https://figshare.com/articles/dataset/CHOLAN-EL-Dataset/13607282)) to download the dataset used in our experiments

## Wikipedia 

* Dataset - AIDA-CoNLL, we used the dataset from the DCA paper. Please refer to this ([repository](https://github.com/YoungXiyuan/DCA)). 

## Candidate Generation

* FALCON 2.0 - The locally indexed KG items have been used. Please refer to this ([repository](https://github.com/SDM-TIB/falcon2.0)) for the set up using the Wikidata dump.
* ([DCA](https://github.com/YoungXiyuan/DCA)) - A predefined candidate set has been used. (Wikipedia)

## Setup 
Requirements: Python 3.6 or 3.7, torch>=1.2.0

## Running 
python cholan.py &nbsp;  

## Citation
```
@inproceedings{kannan-ravi-etal-2021-cholan,
    title = {CHOLAN: A Modular Approach for Neural Entity Linking on Wikipedia and Wikidata},
    author = {Kannan Ravi, Manoj Prabhakar and Singh, Kuldeep and Mulang, Isaiah Onando and Shekarpour, Saeedeh and Hoffart, Johannes and Lehmann, Jens},
    booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
    year = {2021}
}
```
