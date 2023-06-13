import torch
import torch.nn as nn

from utils.train_helpers import *
from utils.rotation import rotate_batch, rotate_single_with_label

def trerr_single(model, image):
	model.eval()
	labels = torch.LongTensor([0, 1, 2, 3])
	inputs = []
	for label in labels:
		inputs.append(rotate_single_with_label(rotation_te_transforms(image), label))
	inputs = torch.stack(inputs)
	inputs, labels = inputs.cuda(), labels.cuda()
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)
	return predicted.eq(labels).cpu()

def adapt_single(model, image, optimizer, criterion, niter, batch_size):
	#breakpoint()
	model.train()
	for iteration in range(niter):
		inputs = [rotation_tr_transforms(image) for _ in range(batch_size)] # 16 (16 rotations)
		inputs, labels = rotate_batch(inputs) # torch.Size([16, 3, 224, 224]), torch.Size([16])
		inputs, labels = inputs.cuda(), labels.cuda()
		optimizer.zero_grad()
		outputs = model(inputs) # torch.Size([16, 4]
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

def adapt_multiple(model, images, optimizer, criterion, niter, batch_size):
	model.train()
	n_examples = len(images)
	for iteration in range(niter):
		inputs = []
		for i in range(n_examples):
			rotations_per_img = [rotation_tr_transforms(images[i]) for _ in range(batch_size)]
			inputs = inputs + rotations_per_img
		inputs, labels = rotate_batch(inputs)
		inputs, labels = inputs.cuda(), labels.cuda() # torch.Size([64, 3, 224, 224]), torch.Size([64]) --> stacked inputs to get size: 16*4, 3, 224, 224
		
		optimizer.zero_grad()
		outputs = model(inputs) # torch.Size([64, 4])
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

def adapt_single_nm(model, image, optimizer, criterion, niter, batch_size):
	#breakpoint()
	model.train()
	for iteration in range(niter):
		breakpoint()
		inputs = [rotation_tr_transforms(image) for _ in range(batch_size)]
		inputs, labels = rotate_batch(inputs)
		inputs, labels = inputs.cuda(), labels.cuda()
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

def test_single(model, image, label):
	model.eval()
	# inputs = te_transforms(image).unsqueeze(0)
	inputs = image.unsqueeze(0)
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)

		confidence = nn.functional.softmax(outputs, dim=1).squeeze()[label].item()

	correctness = 1 if predicted.item() == label else 0
	return correctness, confidence, predicted.item()

def test_multiple(model, images, labels):
	model.eval()
	inputs = images # torch.Size([16, 3, 224, 224])
	with torch.no_grad():
		outputs = model(inputs) # torch.Size([16, 4])
		_, predicted = outputs.max(1)

		confidences = torch.nn.functional.softmax(outputs, dim=1) # orch.Size([16, 4])
		correct_predictions = predicted.eq(labels.cuda()) # torch.Size([16])

	return correct_predictions, confidences, predicted


# FOR DEBUGGING
def test_single_nm(model, image, label):
	model.eval()
	# inputs = te_transforms(image).unsqueeze(0)
	inputs = image.unsqueeze(0).cuda()
	with torch.no_grad():
		outputs = model(inputs.cuda())
		_, predicted = outputs.max(1)

		confidence = nn.functional.softmax(outputs, dim=1).squeeze()[label].item()

	correctness = 1 if predicted.item() == label else 0
	return correctness, confidence, predicted.item()
	
