import torch
import torch.nn as nn

# To save the model, model.state_dict() holds the parameters
torch.save(model.state_dict(), PATH)

# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

