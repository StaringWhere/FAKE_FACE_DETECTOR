import sys
sys.path.insert(0,'.')
import torch
from FeatherNet import FeatherNetB
import torchvision.transforms as transforms

def process(image):
	normalize = transforms.Normalize(mean=[0.14300402, 0.1434545, 0.14277956], 
									 std=[0.10050353, 0.100842826, 0.10034215])
	transform = transforms.Compose([
		transforms.Resize(int(256)),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize,
	])
	image = transform(image).unsqueeze(0)
	return image

def check_spoofing(image, isIR=False):
	image = process(image)
	device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
	model = FeatherNetB()
	if isIR==True:
		model_path = './checkpoints/FeatherNetB_bs32-ir/_54.pth.tar'
	else:
		model_path = './checkpoints/FeatherNetB_bs32/_47_best.pth.tar'
	if torch.cuda.is_available():
		checkpoint = torch.load(model_path)
	else:
		checkpoint = torch.load(model_path,map_location = 'cpu')
	model_dict = {}
	state_dict = model.state_dict()
	for (k,v) in checkpoint['state_dict'].items():
		if k[7:] in state_dict:
			model_dict[k[7:]] = v
	state_dict.update(model_dict)
	model.load_state_dict(state_dict)
	model.eval()
	image.to(device)
	output = model(image)
	soft_output = torch.softmax(output, dim=-1)
	preds = soft_output.to('cpu').detach().numpy()
	return preds[0][0]
	'''
	_, predicted = torch.max(soft_output.data, 1)
	predicted = predicted.to('cpu').detach().numpy()
	if(predicted[0]==1):
		print(image_path+": REAL!")
	else:
		print(image_path + ": FAKE!")
	'''

def check(depth_img, ir_img):
	fake_prob1 = check_spoofing(depth_img)
	fake_prob2 = check_spoofing(ir_img, isIR=True)
	print(fake_prob1, fake_prob2)
	if(fake_prob1>=0.7):
		return 0
	elif(fake_prob1<=0.3):
		return 1
	else:
		if(fake_prob2>0.5):
			return 0
		else:
			return 1
		
