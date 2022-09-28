THIS REPO CONTAINS THE SOURCE CODE OF THE MODELS PRESENTED IN

# Multimodal data fusion of adult and pediatric brain tumors with deep learning

Sandra Steyaert <sup>1,\*</sup>, Yeping Lina Qiu<sup>1,2,\*</sup>, Yuanning Zheng<sup>1</sup>, Pritam Mukherjee<sup>1</sup>, Hannes Vogel<sup>3</sup> and Olivier Gevaert<sup>1,4,5,\*\*</sup>

1)	Stanford Center for Biomedical Informatics Research (BMIR), Department of Medicine, Stanford University, Stanford, CA, USA. 
2)	Department of Electrical Engineering, Stanford University, Stanford, CA, USA. 
3)	Department of Pathology, Stanford University, Stanford, CA, USA.
4)	Department of Biomedical Data Science, Stanford University, Stanford, CA, USA. 
5)	On behalf of The Children’s Brain Tumor Tissue Consortium (CBTTC).

\* Contributed equally

\*\* To whom correspondence should be addressed: ogevaert@stanford.edu


### Abstract

The introduction of deep learning in both imaging and genomics has significantly advanced the analysis of biomedical data. For complex diseases such as cancer different data modalities may reveal different disease characteristics, and the integration of imaging with genomic data has the potential to unravel additional information then when using these data sources in isolation. Here, we propose a DL framework that by combining histopathology images with gene expression profiles can predict prognosis of brain tumors.  Using two separate cohorts of 783 adult and 305 pediatric brain tumors, the developed multimodal data models achieved better prediction results compared to the single data models, but also leads to the identification of more relevant biological pathways. Importantly, when testing our adult models on a third independent brain tumor dataset, we show our multimodal framework is able to generalize and performs better on new data from different cohorts. Furthermore, leveraging the concept of transfer learning, we demonstrate how our multimodal models pre-trained on pediatric glioma can be used to predict prognosis for two more rare (less available samples) pediatric brain tumors, i.e. ependymoma and medulloblastoma. To summarize, our study illustrates that a multimodal data fusion approach can be successfully implemented and customized to model clinical outcome of adult and pediatric brain tumors.
 
