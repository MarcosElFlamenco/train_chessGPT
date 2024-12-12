model_name = "random16M_8layer_6K.pth"

model2 = model_name.split('.')[0]
model_split = model2.split('_')
model_iters = model_split[-1]
model_name = model2[:-(len(model_iters) + 1)]
#dataset = dataset_name.split('.')[0]

print(model_iters, model_name)