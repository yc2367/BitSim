from sim.stripes import Stripes 
from model_profile.models.models import MODEL

model = MODEL['mobilenet_v2']
model = model()

if __name__ == "__main__":
    acc = Stripes(8, 8, [32, 16],'mobilenet_v2', model)
    acc.calc_cycle()
    print(acc.get_pe_array_dim(), '\n', acc.pe.get_energy())
    print(acc.pe_array.dimension)
    print(acc.w_sram.r_cost, acc.w_sram.w_cost)
    