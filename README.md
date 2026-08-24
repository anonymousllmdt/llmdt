# LLMDT: LLM-Based Dual-Transformer for Scene-Aware Text-Driven 3D Human Motion Generation

<img style="max-width: 100%;" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/wax.png" alt="VERHM Overview">

## Overview

We propose LLMDT, an LLM-based dual-transformer framework for text-driven 3D human motion generation in dynamic scenes. By combining LLM-based spatial reasoning, object-centric scene understanding, trajectory planning, and a pose-aware dual-transformer with hierarchical RVQ, the framework generates realistic, collision-free, and context-aware human motions that outperform existing state-of-the-art methods on the HUMANISE dataset.

## Architecture

This section presents the proposed LLM-Based Dual-Transformer (LLMDT) framework for scene-aware text-driven 3D human motion generation. The framework first employs an LLM to extract structured semantic information and ground interaction-relevant objects within the 3D scene. An object-centric representation is then constructed to encode local geometric context, followed by a trajectory transformer that predicts collision-free navigation paths. A trajectory sensor captures dynamic scene occupancy along the predicted path, while a pose-aware dual transformer generates realistic and temporally consistent human motions through hierarchical coarse-to-fine motion refinement.

<img style="max-width: 100%;" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/overview.png" alt="llmdt Overview">

## Installation

```
conda create python=3.9 --name llmdt
conda activate llmdt
```
Install the requirements
```
pip install -r requirements.txt
```
## Demo

Qualitative results demonstrating llmdt's capability to synthesize human movement in 3D scenes from textual descriptions.

We provide the generated visualization results for convenience.

1. Download the generated results from the provided link.
2. Extract the downloaded archive into the directory.
3. You can replace these files with your own generated results.

### Prepare Visualization

We use the **Wis3D** library to visualize the generated motions. First, generate the visualization files by running:

```bash
python visual.py -c configs/test/visual.yaml
```

Open your web browser and navigate to:

```text
http://${HOST}:${PORT}
```

You can now interactively explore the generated results in the Wis3D viewer.

<table>
  <tr>
    <td style="text-align: center;">
      <p>Lie on the bed.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/1.gif" alt="8">
    </td>
    <td style="text-align: center;">
      <p>Sit on the toilet.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/2.gif" alt="787">
    </td>
    <td style="text-align: center;">
      <p>Sit on the chair.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/3.gif" alt="825">
    </td>
      <td style="text-align: center;">
      <p>Sit on the sofa chair that is far away from the tv.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/4.gif" alt="843">
    </td>
  </tr>
    <tr>
    <td style="text-align: center;">
      <p>Stand up from the couch that is far from the table.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/5.gif" alt="1446">
    </td>
    <td style="text-align: center;">
      <p>Stand up from the toilet.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/6.gif" alt="1599">
    </td>
    <td style="text-align: center;">
      <p>Stand up from the bed.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/7.gif" alt="1674">
    </td>
    <td style="text-align: center;">
      <p>Stand up from the armchair that is farthest from the shelf.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/8.gif" alt="1699">
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <p>Walk to the chair that is in the center of the cabinet and the whiteboard.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/9.gif" alt="1856">
    </td>
    <td style="text-align: center;">
      <p>Stand up from the chair that is far away from the door.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/10.gif" alt="1799">
    </td>
    <td style="text-align: center;">
      <p>Walk to the chair that is far from the cabinets.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/11.gif" alt="1899">
    </td>
    <td style="text-align: center;">
      <p>Sit on the chair that is near the ottoman.</p>
      <img width="165" src="https://github.com/anonymousllmdt/llmdt/blob/main/resources/12.gif" alt="699">
    </td>
  </tr>
</table>

## Data Preparation

### ScanNet Dataset

Download the **ScanNet v2** dataset.

- `*_vh_clean_2.ply`
- `*_vh_clean.aggregation.json`
- `*_vh_clean_2*segs.json`

Create a symbolic link to the ScanNet dataset directory:

```bash
mkdir -p data
ln -s /path/to/scannet data/ScanNet
```

Preprocess the ScanNet data using:

```bash
python preprocess_scannet.py
```

The processed files will be saved to:

```text
data/scannet_preprocess
```

### HUMANISE Dataset

Download the **HUMANISE** dataset from the provided link.

Create a symbolic link to the dataset directory:

```bash
mkdir -p data
ln -s /path/to/humanise data/HUMANISE
```

### AMASS Dataset

Please follow the preprocessing pipeline provided by **HUMOR** to download and preprocess the AMASS dataset.

After preprocessing, create a symbolic link:

```bash
ln -s /path/to/amass_processed data/amass_preprocess
```

### SMPL-X Models

Download the **SMPL-X** models.

Place the downloaded `smplx` folder under `data/smpl_models`:

```bash
mkdir -p data/smpl_models
mv smplx data/smpl_models/
```

The final directory structure should be:

```text
data/
├── smpl_models/
│   └── smplx/
```

## Training on the AMASS Dataset

The models are trained on the AMASS dataset to learn general human motion and trajectory priors.

### Train the Trajectory Model

We trained the trajectory model as initialization:

```bash
python train.py -c configs/train/trajgen/traj_amass.yaml task amass_traj
```

### Train the Motion Model

We trained the motion model as initialization:

```bash
python train.py -c configs/train/motiongen/motion_amass.yaml task amass_motion
```

The training logs, checkpoints, and generated models will be saved under:

```text
checkpoints/train/
```

After successful training, the pretrained models will be available at:

```text
checkpoints/train/amass_traj/model
checkpoints/train/amass_motion/model
```

## Training on the HUMANISE Dataset

The models are trained on the HUMANISE dataset for scene-aware motion generation.

### Train the Trajectory Model

We trained the trajectory model as initialization:

```bash
python train.py \
    -c configs/train/trajgen/traj_humanise.yaml \
    task humanise_traj \
    resume True \
    resume_model_dir out/train/amass_traj/model
```

### Train the Motion Model

We trained the motion model as initialization:

```bash
python train.py \
    -c configs/train/motiongen/motion_humanise.yaml \
    task humanise_motion \
    resume True \
    resume_model_dir out/train/amass_motion/model
```

The training logs, checkpoints, and generated models will be saved under:

```text
checkpoints/train/
```

After successful training, the pretrained models will be available at:

```text
checkpoints/train/amass_traj/model
checkpoints/train/amass_motion/model
```
