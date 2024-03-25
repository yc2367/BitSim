from sim.stripes import Stripes 
from model_profile.models.models import MODEL

model = MODEL['resnet50']
model = model()

if __name__ == "__main__":
    acc = Stripes(5, 8, [32, 16],'resnet50', model)
    
    print(acc.get_pe_array_dim(), '\n', acc.pe.get_energy())
    print(acc.input_dim)
    print(acc.output_dim)
    print(acc.weight_dim)
    print(acc.pe_array.dimension)
    print(acc.w_sram.r_cost, acc.w_sram.w_cost)
    