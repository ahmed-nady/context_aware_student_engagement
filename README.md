## Context-aware-Student-Engagement
This repo is the official implementation for **Context Matters: Peer-Aware Student Behavioral Engagement Measurement via VLM Action Parsing and LLM Sequence Classification**, which is accepted at **IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) 2026**.
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2601.06394) <br />
 
## Abstract
Understanding student behavior in the classroom is essential to improve both pedagogical quality and student engagement.
Existing methods for predicting student engagement typically require substantial annotated data to model the diversity of student behaviors, yet privacy concerns often restrict researchers to their own proprietary datasets.
 Moreover, the classroom context, represented in peers' actions, is ignored. To address the aforementioned limitation, we propose a novel three-stage framework for video-based student engagement measurement. 
 First, we explore the few-shot adaptation of the vision-language model for student action recognition, which is fine-tuned to distinguish among action categories with a few training samples. 
 Second, to handle continuous and unpredictable student actions, we utilize the sliding temporal window technique to divide each student's 2-minute-long video into non-overlapping segments.
 Each segment is assigned an action category via the fine-tuned VLM model, generating a sequence of action predictions. Finally, we leverage the large language model to classify this entire sequence of actions, together with the classroom context, as belonging to an engaged or disengaged student. 
The experimental results demonstrate the effectiveness of the proposed approach in identifying student engagement.

# Architecture of VLM-LLM Framework
<div align=center>
<img src ="./figures/proposedFramework.png" width="1000"/>
</div>

# Prerequisites
- Python3
- [PyTorch](http://pytorch.org/)
- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt `
# Testing Pretrained Models
You may download the trained models reported in the paper via [GoogleDrive](https://drive.google.com/drive/folders/1yBpdM3Wt6F69HoXHpqGB-XkcItsXu6H3?usp=drive_link) and put them in folder pretrained_action_recog_models.

# Single student Demo

You can use the following command to test on a single student video clip. 
```shell
python SingleStudentDemo.py
```
# multiple students Demo
You can use the following command to test on video of multiple students, which requires detection and tracking.
```shell
python MultipleStudentsDemo.py
```
Note:  use engageConfig.py to set videos paths, etc.
# Citation
Please cite this work if you find it useful:
```BibTex
@article{abdelkawy2026context,
  title={Context Matters: Peer-Aware Student Behavioral Engagement Measurement via VLM Action Parsing and LLM Sequence Classification},
  author={Abdelkawy, Ahmed and Elsayed, Ahmed and Ali, Asem and Farag, Aly and Tretter, Thomas and McIntyre, Michael},
  journal={arXiv preprint arXiv:2601.06394},
  year={2026}
}
 
