import torch.nn as nn
import torch



def mark_adapter_trainable_params(model):
    for name, param in model.named_parameters():
        if 'adapter' not in name:
            param.requires_grad = False
        else:
            print("training: {}".format(name))

def mark_lora_trainable_params(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            print("training: {}".format(n))



def load_state_dict(model, ckp_path):
    model_dict = model.state_dict()
    my_model_unload_key = model.state_dict()
    pretrained_dict = torch.load(ckp_path)
    predtrained_dict_unload = torch.load(ckp_path)


    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
            del my_model_unload_key[k]
            del predtrained_dict_unload[k]

        else:
            no_load_key.append(k)

    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)

    # ------------------------------------------------------#
    #   显示没有匹配上的Key
    # ------------------------------------------------------#
    print("\nPretrained Key Num", len(pretrained_dict))
    print("\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key Num:", len(no_load_key))
    print("\nMy Model Key Num:", len(model_dict))
    print("\nMy Model Unload Key Num:",  len(my_model_unload_key))
    print("\nMy Model Unload Key:",  my_model_unload_key.keys())
    print("\nPretrained Unload Key Num:", len(predtrained_dict_unload))
    print("\nPretrained Unload Key:", predtrained_dict_unload.keys())

