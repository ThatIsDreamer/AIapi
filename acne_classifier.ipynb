{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2025-05-19T17:54:40.889713Z",
     "iopub.status.busy": "2025-05-19T17:54:40.889434Z",
     "iopub.status.idle": "2025-05-19T17:54:40.895307Z",
     "shell.execute_reply": "2025-05-19T17:54:40.894568Z",
     "shell.execute_reply.started": "2025-05-19T17:54:40.889694Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "import PIL\n",
    "import torchvision\n",
    "from PIL import Image\n",
    "from torchvision.transforms import v2\n",
    "\n",
    "import os\n",
    "import shutil\n",
    "import random\n",
    "\n",
    "import torchvision.transforms.functional as TF\n",
    "from PIL import Image\n",
    "from torchvision.transforms import CenterCrop, Compose, Pad, Resize, ToTensor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T16:47:06.881320Z",
     "iopub.status.busy": "2025-05-19T16:47:06.880772Z",
     "iopub.status.idle": "2025-05-19T16:47:26.725109Z",
     "shell.execute_reply": "2025-05-19T16:47:26.724321Z",
     "shell.execute_reply.started": "2025-05-19T16:47:06.881299Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "folder1 = \"/kaggle/input/acne-dataset/Acne\"\n",
    "folder2 = \"/kaggle/input/oily-dry-and-normal-skin-types-dataset/Oily-Dry-Skin-Types/train/normal\"\n",
    "\n",
    "output_folder_normal = (\n",
    "    \"combined_dataset_acne_final/normal\"\n",
    ")\n",
    "output_folder_acne = \"combined_dataset_acne_final/acne\"\n",
    "\n",
    "# Создаём общую папку (если её нет)\n",
    "os.makedirs(output_folder_normal, exist_ok=True)\n",
    "os.makedirs(output_folder_acne, exist_ok=True)\n",
    "\n",
    "for filename in os.listdir(folder1):\n",
    "    src = os.path.join(folder1, filename)\n",
    "    dst = os.path.join(output_folder_acne, filename)\n",
    "    if os.path.isfile(src): \n",
    "        shutil.copy(src, dst)\n",
    "\n",
    "for filename in os.listdir(folder2):\n",
    "    src = os.path.join(folder2, filename)\n",
    "    dst = os.path.join(output_folder_normal, filename)\n",
    "    if os.path.isfile(src): \n",
    "        shutil.copy(src, dst)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T18:25:11.233880Z",
     "iopub.status.busy": "2025-05-19T18:25:11.233189Z",
     "iopub.status.idle": "2025-05-19T18:25:11.239205Z",
     "shell.execute_reply": "2025-05-19T18:25:11.238330Z",
     "shell.execute_reply.started": "2025-05-19T18:25:11.233837Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "def resize_with_padding(image, target_size=(214, 214)):\n",
    "    original_size = image.size\n",
    "    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])\n",
    "    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))\n",
    "    image = image.resize(new_size, Image.Resampling.LANCZOS)\n",
    "\n",
    "    new_img = Image.new(\"RGB\", target_size, (0, 0, 0))  \n",
    "    paste_position = (\n",
    "        (target_size[0] - new_size[0]) // 2,\n",
    "        (target_size[1] - new_size[1]) // 2,\n",
    "    )\n",
    "    new_img.paste(image, paste_position)\n",
    "    return new_img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T17:56:13.683719Z",
     "iopub.status.busy": "2025-05-19T17:56:13.683467Z",
     "iopub.status.idle": "2025-05-19T17:56:27.301733Z",
     "shell.execute_reply": "2025-05-19T17:56:27.301019Z",
     "shell.execute_reply.started": "2025-05-19T17:56:13.683702Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████| 476/476 [00:08<00:00, 57.23it/s]\n",
      "100%|██████████████████████████████████████| 1162/1162 [00:05<00:00, 219.82it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Всего картинок: 1026\n"
     ]
    }
   ],
   "source": [
    "from os import listdir\n",
    "from os.path import isfile, join\n",
    "\n",
    "\n",
    "from tqdm import tqdm\n",
    "\n",
    "output_folder = \"/kaggle/input/acne-data/Acne_data\"\n",
    "\n",
    "images_source = []\n",
    "s = 0\n",
    "for target, class_name in enumerate([\"acne\", \"not_acne\"]):\n",
    "    if class_name == \"acne\":\n",
    "        class_folder = f\"{output_folder}/{class_name}\"\n",
    "        for image_name in tqdm(listdir(f\"{output_folder}/{class_name}\"), ncols=80):\n",
    "            image_path = f\"{class_folder}/{image_name}\"        \n",
    "            if not isfile(image_path):\n",
    "                continue\n",
    "            if not image_name.lower().endswith((\".jpg\", \".jpeg\", \".png\", \".bmp\", \".gif\", \"JPG\")):\n",
    "                continue\n",
    "\n",
    "            try:\n",
    "                with Image.open(image_path) as img:\n",
    "                    img = resize_with_padding(img, (214, 214))\n",
    "                    images_source.append((img, target))\n",
    "            except Exception as e:\n",
    "                print(f\"Ошибка при открытии {image_path}: {e}\")\n",
    "    else:\n",
    "        class_folder = f\"{output_folder}/{class_name}\"\n",
    "        for image_name in tqdm(listdir(f\"{output_folder}/{class_name}\"), ncols=80):\n",
    "            if s < 550:\n",
    "                image_path = f\"{class_folder}/{image_name}\"        \n",
    "                if not isfile(image_path):\n",
    "                    continue\n",
    "                if not image_name.lower().endswith((\".jpg\", \".jpeg\", \".png\", \".bmp\", \".gif\")):\n",
    "                    continue\n",
    "\n",
    "                try:\n",
    "                    with Image.open(image_path) as img:\n",
    "                        img = resize_with_padding(img, (214, 214))\n",
    "                        images_source.append((img, target))\n",
    "                        s += 1\n",
    "                except Exception as e:\n",
    "                    print(f\"Ошибка при открытии {image_path}: {e}\")\n",
    "\n",
    "print(\"\\nВсего картинок:\", len(images_source))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T17:01:16.465636Z",
     "iopub.status.busy": "2025-05-19T17:01:16.465294Z",
     "iopub.status.idle": "2025-05-19T17:01:25.517543Z",
     "shell.execute_reply": "2025-05-19T17:01:25.516929Z",
     "shell.execute_reply.started": "2025-05-19T17:01:16.465598Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import zipfile\n",
    "\n",
    "def zip_folder(folder_path, output_zip_path):\n",
    "    with zipfile.ZipFile(output_zip_path, \"w\", zipfile.ZIP_DEFLATED) as zipf:\n",
    "        for root, dirs, files in os.walk(folder_path):\n",
    "            for file in files:\n",
    "                abs_path = os.path.join(root, file)\n",
    "                rel_path = os.path.relpath(abs_path, folder_path)\n",
    "                zipf.write(abs_path, rel_path)\n",
    "\n",
    "zip_folder(\"combined_dataset_acne_final\", \"combined_dataset_acne_final.zip\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T17:59:36.712065Z",
     "iopub.status.busy": "2025-05-19T17:59:36.711484Z",
     "iopub.status.idle": "2025-05-19T17:59:36.716221Z",
     "shell.execute_reply": "2025-05-19T17:59:36.715595Z",
     "shell.execute_reply.started": "2025-05-19T17:59:36.712039Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from random import shuffle\n",
    "\n",
    "shuffle(images_source)\n",
    "train_images_source = images_source[:800]\n",
    "test_images_source = images_source[801:]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T18:52:09.615821Z",
     "iopub.status.busy": "2025-05-19T18:52:09.615464Z",
     "iopub.status.idle": "2025-05-19T18:52:09.623096Z",
     "shell.execute_reply": "2025-05-19T18:52:09.622390Z",
     "shell.execute_reply.started": "2025-05-19T18:52:09.615757Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from torch.utils.data import DataLoader, Dataset\n",
    "from torchvision import transforms\n",
    "\n",
    "\n",
    "class TnJDataset(Dataset):\n",
    "    def __init__(self, source, is_train=True):\n",
    "        self.is_train = is_train\n",
    "        self.test_transform = transforms.Compose(\n",
    "            [\n",
    "                transforms.ToTensor(),\n",
    "                transforms.Normalize(\n",
    "                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]\n",
    "                ),\n",
    "            ]\n",
    "        )\n",
    "        self.train_transform = transforms.Compose(\n",
    "            [\n",
    "                transforms.RandomHorizontalFlip(p=0.5),\n",
    "                transforms.RandomRotation(degrees=10),\n",
    "                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.5)),\n",
    "                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),\n",
    "                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),\n",
    "                transforms.RandomResizedCrop(214, scale=(0.9, 1.0)),\n",
    "                transforms.ToTensor(),\n",
    "                transforms.Normalize(\n",
    "                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]\n",
    "                ),\n",
    "            ]\n",
    "        )\n",
    "        data, target = list(zip(*source))\n",
    "        self.data = data\n",
    "        self.target = torch.tensor(target)\n",
    "\n",
    "    def __getitem__(self, index):\n",
    "        image, label = self.data[index], self.target[index]\n",
    "        if self.is_train:\n",
    "            image = self.train_transform(image)\n",
    "        else:\n",
    "            image = self.test_transform(image)\n",
    "        return image, label\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T17:59:40.792561Z",
     "iopub.status.busy": "2025-05-19T17:59:40.791739Z",
     "iopub.status.idle": "2025-05-19T17:59:40.799714Z",
     "shell.execute_reply": "2025-05-19T17:59:40.798808Z",
     "shell.execute_reply.started": "2025-05-19T17:59:40.792517Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "dataset_train = TnJDataset(train_images_source, is_train=True)\n",
    "dataset_test = TnJDataset(test_images_source, is_train=False)\n",
    "\n",
    "dataloader_train = DataLoader(\n",
    "    dataset=dataset_train,\n",
    "    batch_size=128,\n",
    "    shuffle=True,\n",
    ")\n",
    "dataloader_test = DataLoader(\n",
    "    dataset=dataset_test,\n",
    "    batch_size=512,\n",
    "    shuffle=False,\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T17:59:43.727342Z",
     "iopub.status.busy": "2025-05-19T17:59:43.726442Z",
     "iopub.status.idle": "2025-05-19T17:59:43.743278Z",
     "shell.execute_reply": "2025-05-19T17:59:43.742303Z",
     "shell.execute_reply.started": "2025-05-19T17:59:43.727306Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "import torch.optim as optim\n",
    "from torchvision import datasets, models, transforms\n",
    "\n",
    "accuracy_arr = []\n",
    "\n",
    "\n",
    "def train_model(train_loader, test_loader, n_epoch=16):\n",
    "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "    model = models.efficientnet_b0(pretrained=True)\n",
    "\n",
    "    for name, param in model.named_parameters():\n",
    "        if \"layer4\" in name or \"fc\" in name:\n",
    "            param.requires_grad = True\n",
    "        else:\n",
    "            param.requires_grad = False\n",
    "\n",
    "\n",
    "    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)\n",
    "\n",
    "    model = model.to(device)\n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)\n",
    "    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)\n",
    "\n",
    "\n",
    "    for epoch in range(n_epoch):\n",
    "        model.train()\n",
    "        running_loss = 0.0\n",
    "        correct = 0\n",
    "        total = 0\n",
    "\n",
    "        for X, target in tqdm(train_loader, ncols=80):\n",
    "            X, target = X.to(device), target.to(device)\n",
    "            optimizer.zero_grad()\n",
    "\n",
    "            logits = model(X)\n",
    "            loss = criterion(logits, target)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "\n",
    "            running_loss += loss.item() * X.size(0)\n",
    "            _, preds = torch.max(logits, 1)\n",
    "            correct += torch.sum(preds == target).item()\n",
    "            total += target.size(0)\n",
    "\n",
    "        epoch_loss = running_loss / total\n",
    "        epoch_acc = correct / total\n",
    "        accuracy_arr.append(epoch_acc)\n",
    "        print(\n",
    "            f\"Epoch {epoch+1}/{n_epoch} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}\"\n",
    "        )\n",
    "        scheduler.step()\n",
    "\n",
    "    model.eval()\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    with torch.no_grad():\n",
    "        for X, target in tqdm(test_loader, ncols=80):\n",
    "            X, target = X.to(device), target.to(device)\n",
    "            logits = model(X)\n",
    "            _, preds = torch.max(logits, 1)\n",
    "            correct += torch.sum(preds == target).item()\n",
    "            total += target.size(0)\n",
    "\n",
    "    val_acc = correct / total\n",
    "    print(f\"Validation Accuracy: {val_acc:.4f}\")\n",
    "    return model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T17:59:48.111825Z",
     "iopub.status.busy": "2025-05-19T17:59:48.111469Z",
     "iopub.status.idle": "2025-05-19T18:02:41.369030Z",
     "shell.execute_reply": "2025-05-19T18:02:41.368156Z",
     "shell.execute_reply.started": "2025-05-19T17:59:48.111793Z"
    },
    "trusted": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.\n",
      "  warnings.warn(\n",
      "/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=EfficientNet_B0_Weights.IMAGENET1K_V1`. You can also use `weights=EfficientNet_B0_Weights.DEFAULT` to get the most up-to-date weights.\n",
      "  warnings.warn(msg)\n",
      "Downloading: \"https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth\" to /root/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth\n",
      "100%|██████████| 20.5M/20.5M [00:00<00:00, 125MB/s] \n",
      "100%|█████████████████████████████████████████████| 7/7 [00:11<00:00,  1.71s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/16 - Loss: 0.6930 - Acc: 0.5400\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.52s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 2/16 - Loss: 0.6579 - Acc: 0.6338\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.52s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 3/16 - Loss: 0.6356 - Acc: 0.6787\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.50s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 4/16 - Loss: 0.5982 - Acc: 0.7300\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.51s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 5/16 - Loss: 0.5642 - Acc: 0.7538\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.53s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 6/16 - Loss: 0.5423 - Acc: 0.7887\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.52s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 7/16 - Loss: 0.5131 - Acc: 0.7975\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.54s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 8/16 - Loss: 0.4934 - Acc: 0.8087\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.52s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 9/16 - Loss: 0.4643 - Acc: 0.8263\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.52s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 10/16 - Loss: 0.4389 - Acc: 0.8413\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.53s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 11/16 - Loss: 0.4182 - Acc: 0.8525\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.53s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 12/16 - Loss: 0.4161 - Acc: 0.8387\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.51s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 13/16 - Loss: 0.4008 - Acc: 0.8538\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.55s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 14/16 - Loss: 0.3935 - Acc: 0.8375\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.52s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 15/16 - Loss: 0.3879 - Acc: 0.8413\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 7/7 [00:10<00:00,  1.54s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 16/16 - Loss: 0.3827 - Acc: 0.8512\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|█████████████████████████████████████████████| 1/1 [00:00<00:00,  2.01it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation Accuracy: 0.8400\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "resnet_model = train_model(dataloader_train, dataloader_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T18:04:00.706073Z",
     "iopub.status.busy": "2025-05-19T18:04:00.705319Z",
     "iopub.status.idle": "2025-05-19T18:04:00.723065Z",
     "shell.execute_reply": "2025-05-19T18:04:00.722144Z",
     "shell.execute_reply.started": "2025-05-19T18:04:00.706045Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "from PIL import Image\n",
    "from torchvision import transforms\n",
    "\n",
    "def prediction_final(model1, image_arr):\n",
    "    transform = transforms.Compose(\n",
    "        [\n",
    "            transforms.ToTensor(),\n",
    "            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),\n",
    "        ]\n",
    "    )\n",
    "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "    \n",
    "    acne_arr = []\n",
    "    for i in image_arr:\n",
    "        img = Image.open(i).convert(\"RGB\")\n",
    "        img = resize_with_padding(img, (214, 214))\n",
    "        img_tensor = transform(img).unsqueeze(0)\n",
    "\n",
    "        model1.eval()\n",
    "        model1 = model1.to(device)\n",
    "\n",
    "        img_tensor = img_tensor.to(device)\n",
    "\n",
    "        with torch.no_grad():\n",
    "            logits = model1(img_tensor)\n",
    "            probs = torch.softmax(logits, dim=1)\n",
    "            print(probs)\n",
    "            predicted_class = torch.argmax(probs, dim=1).item()\n",
    "            if abs(probs[0][0] - probs[0][1]) < 0.15:\n",
    "                predicted_class = 2\n",
    "            elif 0.15 < abs(probs[0][0] - probs[0][1]) < 0.3:\n",
    "                predicted_class = 0\n",
    "\n",
    "        class_names = [\"Acne\", \"No Acne\", \"Hesitation\"]\n",
    "        acne_arr.append(class_names[predicted_class])\n",
    "        \n",
    "    if acne_arr.count(\"Acne\") >= 1:\n",
    "        return \"Acne\"\n",
    "    if acne_arr.count(\"Hesitation\") >= 1:\n",
    "        return \"Hesitation\"\n",
    "    return \"No Acne\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2025-05-19T18:13:09.260695Z",
     "iopub.status.busy": "2025-05-19T18:13:09.260388Z",
     "iopub.status.idle": "2025-05-19T18:13:09.342548Z",
     "shell.execute_reply": "2025-05-19T18:13:09.341661Z",
     "shell.execute_reply.started": "2025-05-19T18:13:09.260673Z"
    },
    "trusted": true
   },
   "outputs": [],
   "source": [
    "torch.save(resnet_model, \"/kaggle/working/full_resnet_model.pth\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "trusted": true
   },
   "outputs": [],
   "source": [
    "# print(prediction_final(resnet_model, [\"testskin/masha.jpg\", \"testskin/masha1.jpg\", \"testskin/masha2.jpg\"]))\n",
    "# print(prediction_final(resnet_model, [\"testskin/natasha.jpg\", \"testskin/natasha1.jpg\", \"testskin/natasha2.jpg\"]))\n",
    "# print(prediction_final(resnet_model, [\"testskin/1.jpg\", \"testskin/2.jpg\", \"testskin/3.jpg\"]))\n",
    "# print(prediction_final(resnet_model, [\"testskin/4.jpg\", \"testskin/5.jpg\", \"testskin/6.jpg\"]))\n",
    "# print(prediction_final(resnet_model, [\"testskin/yarik.jpg\", \"testskin/yarik1.jpg\", \"testskin/yarik2.jpg\"]))"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "nvidiaTeslaT4",
   "dataSources": [
    {
     "datasetId": 2553733,
     "sourceId": 4337255,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 4470475,
     "sourceId": 7665644,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 7462730,
     "sourceId": 11874666,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 31041,
   "isGpuEnabled": true,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
