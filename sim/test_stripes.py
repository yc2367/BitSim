from sim.stripes import Stripes 

name_list = ['resnet34', 'resnet50', 'vit-small', 'vit-base']

if __name__ == "__main__":
    for i in range(0, 1):
        name = name_list[i]
        acc = Stripes(8, 8, 16, [16, 16], name, None)

        total_cycle    = acc.calc_cycle()
        compute_energy = acc.calc_compute_energy() / 1e6
        sram_rd_energy = acc.calc_sram_rd_energy() / 1e6
        sram_wr_energy = acc.calc_sram_wr_energy() / 1e6
        dram_energy    = acc.calc_dram_energy() / 1e6
        onchip_energy  = compute_energy + sram_rd_energy + sram_wr_energy
        total_energy   = compute_energy + sram_rd_energy + sram_wr_energy + dram_energy

        print(f'model: {name}')
        print(f'total cycle:        {total_cycle}')
        print(f'weight buffer area: {acc.w_sram.area} mm2')
        print(f'input buffer area:  {acc.i_sram.area} mm2')
        print(f'compute energy:     {compute_energy} uJ')
        print(f'sram rd energy:     {sram_rd_energy} uJ')
        print(f'sram wr energy:     {sram_wr_energy} uJ')
        print(f'dram energy:        {dram_energy} uJ')
        print(f'on-chip energy:     {onchip_energy} uJ')
        print(f'total energy:       {total_energy} uJ')

        print()
    