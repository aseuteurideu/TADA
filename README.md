# Targeted Augmented Data for Audio Deepfake Detection
This repository contains the implementation of Audio Deepfake Detection method proposed in the paper -  
  
Marcella Astrid, Enjie Ghorbel, and Djamila Aouada, Targeted Augmented Data for Audio Deepfake Detection, EUSIPCO 2024.  
Links: [[PDF]](https://arxiv.org/abs/2407.07598) 

# Dependencies
Create conda environment with package inside the package list
`conda create -n myenv --file package-list.txt`

# Prepare data
1) Download ASVspoof2019 dataset [here](https://datashare.ed.ac.uk/handle/10283/3336). 
  
2) Store the dataset. Directory paths from the current project directory includes:  

   ```
   ASVspoof2019/LA/ASVspoof2019_LA_asv_protocols/ 
   ASVspoof2019/LA/ASVspoof2019_LA_asv_scores/
   ASVspoof2019/LA/ASVspoof2019_LA_asv_protocols/
   ASVspoof2019/LA/ASVspoof2019_LA_dev/
   ASVspoof2019/LA/ASVspoof2019_LA_eval/
   ASVspoof2019/LA/ASVspoof2019_LA_train/ 
   ```
  
# Training
For rawnet2 training with our method
```
python main.py --atk_prob 0.7 --atk_epsmax 0.7 --atk_epsmin 0.01
```

For aasist training with our method
```
python main.py --atk_prob 0.5 --atk_epsmax 0.5 --atk_epsmin 0.01 --batch_size 16 --model_config config/aasist.yaml
```

For untargeted augmentation (gaussian noise)
```
python main.py --noise_prob 0.7 --noise_std_min 0.01 --noise_std_max 1
```

For targeting confident fake prediction
```
python main.py --atk_prob 0.3 --atk_epsmax 0.7 --atk_epsmin 0.01 --atk_type fake
```

For rawnet2 training without augmentation
```
python main.py 
```

For aasist training without augmentation
```
python main.py --batch_size 16 --model_config config/aasist.yaml
```

# Testing
For rawnet2 with our method ([weight file](https://drive.google.com/drive/folders/1LC7aPFu4ZHoJwf5Kwax5wNLGdcUsSrHF?usp=sharing))
```
python main.py --test log_tmp/v1_LA_100_32_0.0001_atk0.7_ate0.01-0.7_trial1/model/model_best_epoch100.pth.tar
python test.py --folder test_results/v1_LA_100_32_0.0001_atk0.7_ate0.01-0.7_trial1/model_best_epoch100
```

For aasist with our method ([weight file](https://drive.google.com/drive/folders/1zFn9To-NV3DGedLyYFCZoSJZfHdJZW4n?usp=sharing))
```
python main.py --test log_tmp/v1_LA_100_16_0.0001_atk0.5_ate0.01-0.5_aasist/model/model_best_epoch100.pth.tar --batch_size 16 --model_config config/aasist.yaml
python test.py --folder test_results/v1_LA_100_16_0.0001_atk0.5_ate0.01-0.5_aasist/model_best_epoch100
```

Check the results in output file, e.g., for aasist case, at ``test_results/v1_LA_100_16_0.0001_atk0.5_ate0.01-0.5_aasist/model_best_epoch100/output.txt``
  
# Reference
If you use the code, please cite the paper
```
@InProceedings{astrid2024targeted,
  author       = "Astrid, Marcella and Ghorbel, Enjie and Aouada, Djamila",
  title        = "Targeted Augmented Data for Audio Deepfake Detection",
  booktitle    = "32nd European Signal Processing Conference (EUSIPCO)",
  year         = "2024",
}
```
# Acknowledgements
Thanks to the code available at https://github.com/clovaai/aasist, https://github.com/asvspoof-challenge/2021, https://github.com/eurecom-asp/RawGAT-ST-antispoofing.  
  



