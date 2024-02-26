# Theta modulation analysis

This is an application (PostSorting/theta_phase_analysis.py) that runs on the output of the pipeline and requires the raw electrophysiology data and the following data frames: position.pkl, spatial_firing.pkl.

Steps of analysis
1. Filter each channel in the theta range. (using scipy.signal.butter, 5-9Hz, order=5 (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html). 

![image](https://user-images.githubusercontent.com/16649631/154485013-b439f627-d86d-4ce7-9abe-646ddafa1f9a.png)


2. Perform Hilber transform on filtered data. (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html)

![image](https://user-images.githubusercontent.com/16649631/154484870-6afc2839-0d41-4012-9644-9b4e238f7d8f.png)

3. Calculate phase angle and save as a npy array

![image](https://user-images.githubusercontent.com/16649631/154485111-291dda4b-b8cc-4d00-88fa-76272e475f60.png)

![image](https://user-images.githubusercontent.com/16649631/154485191-5e1b9367-c8bb-4dff-8d3d-2d1ae86f522a.png)

4. Load position data (x,y,hd...) and upsample to 120Hz (from 30Hz) by interpolating extra data (pd.interpolate after adding rows of nans).
5. Load theta phase angles and downsample to 120Hz to match the position data and save a combined data frame with the upsampled position data and downsampled theta angles.
6. Load spatial_firing.pkl and find the corresponding theta angles for each spike of each cell and save in spatial_firing_theta.pkl. (Use the theta angle of the primary channel / the channel where the cell had the highest amplitude).
(7.) Save data for an example cell. Load position and spike data for an example cell and save data as a feather file.
To open the feather files in R, something like this should work:
```
install.packages('arrow')
library(arrow)


theta_position <- read_feather(file.path(getwd(), "test_data", "position_theta_cluster_7.feather"))
print(theta_position)
theta_firing <- read_feather(file.path(getwd(), "test_data", "spatial_firing_theta_cluster_7.feather"))
print(theta_firing)
```

![image](https://user-images.githubusercontent.com/16649631/154513473-efc96ee2-a63d-4f20-aba9-a20116ef34e7.png)

![image](https://user-images.githubusercontent.com/16649631/154513566-450d94a2-9b6a-49a2-a788-2b5d6647b87d.png)

