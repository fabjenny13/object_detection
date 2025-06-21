# Urban Vision Hackathon - Phase 2 Scripts

This repository contains data processing and evaluation scripts for the Urban Vision Hackathon Phase 2.

## Scripts Available

### 1. Data Processing Script (`scripts/data_processing_tutorial.py`)
Processes collaborative annotation data from CSV files to generate annotation matrices and user performance metrics.

**Quick Start:**
```bash
# Install requirements
pip install -r requirements.txt

# Basic usage - analyze sample data
python scripts/data_processing_tutorial.py

# Analyze specific image
python scripts/data_processing_tutorial.py --image-id 12345

# Get user accuracy for specific user and image
python scripts/data_processing_tutorial.py --image-id 12345 --user-id 6483 --user-analysis
```

### 2. Evaluation Script (`scripts/evaluation.py`)
Calculates mAP and accuracy metrics for object detection models in COCO format.

**Quick Start:**
```bash
python scripts/evaluation.py ground_truth.json predictions.json
```

## Requirements
```bash
pip install -r requirements.txt
```

## Data Format
The data processing script expects CSV files with:
- `phase_2_image.csv` - Image metadata
- `phase_2_user_annotation.csv` - User annotations  
- `phase_2_user_progression_score.csv` - User performance data
- `phase_2_user_image_user_annotation.csv` - User assignments

---

## LICENSE and COPYRIGHT
* Code in this repository is provided only for use for the duration of the Phase 2 Urban Vision Hackathon being held at IISc on June 21 and 22, 2025. It is covered under the following license.
* Relevant code and datasets will be released in the public domain in the future after due diligence for accuracy, privacy, etc.

```
All Rights Reserved

Copyright (c) 2025 Indian Institute of Science

THE CONTENTS OF THIS REPOSITORY ARE PROPRIETARY AND CONFIDENTIAL.
UNAUTHORIZED COPYING, TRANSFERRING OR REPRODUCTION OF THE CONTENTS OF THIS REPOSITORY, VIA ANY MEDIUM IS STRICTLY PROHIBITED.

The receipt or possession of the source code, datasets, models and/or any parts thereof does not convey or imply any right to use them
for any purpose other than the purpose for which they were provided to you.

The software, datasets, models, etc. are provided "AS IS", without warranty of any kind, express or implied, including but not limited to
the warranties of merchantability, fitness for a particular purpose and non infringement.
In no event shall the authors or copyright holders be liable for any claim, damages or other liability,
whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software, datasets, models, etc.
or the use or other dealings in the software, datasets, models, etc.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the software, datasets, models, etc.
```
