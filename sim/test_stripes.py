from sim.stripes import Stripes 
from model_profile.models.models import MODEL

name_list = ['resnet50', 'mobilenet_v2']
name = name_list[1]
model = MODEL[name]
model = model()

if __name__ == "__main__":
    acc = Stripes(8, 8, 16, [16, 16], name, model)
    print(f'total cycle: {acc.calc_cycle()}')

    compute_energy = acc.calc_compute_energy() / 1e6
    sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
    sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
    dram_energy    = acc.calc_dram_energy() / 1e6
    total_energy = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy
    print(f'compute energy: {compute_energy} uJ')
    print(f'sram rd energy: {sram_rd_energy} uJ')
    print(f'sram wr energy: {sram_wr_energy} uJ')
    print(f'dram energy: {dram_energy} uJ')
    print(f'total energy: {total_energy} uJ')
    