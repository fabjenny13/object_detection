# Creating and running Docker image locally
#### Step 1: Install Docker:
- Follow: https://docs.docker.com/engine/install/

#### Step 2: Download provided files
Clone this repository and make a copy of this folder

#### Step 3: Update main.py with your image annotation model
Note:
- Script output should be in /output directory
- Script image input should be from /input directory
- Example file is provided

#### Step 4: Build Docker image
In the root folder run
```bash
docker build -t annotate_image .
```

#### Step 5: Run and test Docker image
```
docker run -v <input-image-directory>:/input -v <output-directory>:/output annotate_image
```

Ensure that:
- Specified input directory has given image set
- Specified output directory contains generated json file **after successful run**

If all previous steps are successful proceed to the following:

# Uploading Docker image to Docker Hub

In order to evaluate your submission you must upload your final image to Docker Hub
#### Step 6: Create a Repo on Docker Hub
- Go to https://hub.docker.com/
- Click **"Repositories"** > **"New repository"**
- Set the name (e.g., `annotate_image`) and visibility **public**
- Click **"Create"**

#### Step 7: Log in to Docker Hub
```bash
docker login
```

Enter your Docker Hub username and password when prompted.

#### Step 8: Tag your image
Docker images must be tagged using the format:
```
<dockerhub-username>/<repository-name>:<tag>
```

Example:
```
docker tag annotate_image yourusername/annotate_image:latest
```

If you're not sure of your image ID or name, run:
```
docker images
```

#### Step 9: Push the image
```
docker push yourusername/annotate_image:latest
```

Note the repository id (eg: `yourusername/annotate_image:latest`) for your submission
