def _calc_conv2d_cycle(self, w_dim, o_dim):
        pe_group_size = self.PE_GROUP_SIZE
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']
        w_prec = self.pe.input_precision_s

        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, _ = o_dim

        # cycle_kernel:       number of cycles to process a kernel
        # cycle_out_channel:  number of cycles along output channel
        # cycle_out_width:    number of cycles along output width
        # cycle_out_height:   number of cycles along output height
        if cw < pe_group_size:
            cycle_kernel   = math.ceil((k**2) / pe_group_size) * cw
        else:
            cycle_kernel   = math.ceil(cw / pe_group_size) * (k**2)
        cycle_out_channel  = math.ceil(cout / num_pe_row)
        cycle_out_width    = math.ceil(ow / num_pe_col)
        cycle_out_height   = oh

        cycle_per_batch = (cycle_kernel * cycle_out_channel * cycle_out_width * cycle_out_height) * w_prec
        total_cycle = cycle_per_batch * batch_size
        return total_cycle
    
    def _calc_dwconv_cycle(self, w_dim, i_dim, o_dim):
        pe_group_size = self.PE_GROUP_SIZE
        num_pe_col = self.pe_array_dim['w']
        w_prec = self.pe.input_precision_s

        # kernel size, kernel input channel, output channel
        _, _, _, cin = i_dim
        # kernel size, kernel input channel, output channel
        k, _, cw, cout = w_dim
        # batch size, output feature width, output feature height, output channel
        batch_size, ow, oh, _ = o_dim

        assert cin != cw, 'Not a depth-wise convolution!'

        # cycle_kernel:       number of cycles to process a kernel
        # cycle_out_channel:  number of cycles along output channel
        # cycle_out_width:    number of cycles along output width
        # cycle_out_height:   number of cycles along output height
        if cw < pe_group_size:
            cycle_kernel   = math.ceil((k**2) / pe_group_size) * cw
        else:
            cycle_kernel   = math.ceil(cw / pe_group_size) * (k**2)
        cycle_out_channel  = cout
        cycle_out_width    = math.ceil(ow / num_pe_col)
        cycle_out_height   = oh

        cycle_per_batch = (cycle_kernel * cycle_out_channel * cycle_out_width * cycle_out_height) * w_prec
        total_cycle = cycle_per_batch * batch_size
        return total_cycle

    def _calc_fc_cycle(self, w_dim, o_dim):
        pe_group_size = self.PE_GROUP_SIZE
        num_pe_row = self.pe_array_dim['h']
        num_pe_col = self.pe_array_dim['w']
        w_prec = self.pe.input_precision_s

        # kernel size, input channel, output channel
        cin, cout = w_dim
        # batch size, output feature width, output channel
        batch_size, _ = o_dim

        # cycle_in_channel:   number of cycles along input channel
        # cycle_out_channel:  number of cycles along output channel
        cycle_in_channel  = math.ceil(cin / pe_group_size)
        cycle_out_channel = math.ceil(cout / num_pe_row)
        cycle_batch       = math.ceil(batch_size / num_pe_col)

        total_cycle = (cycle_in_channel * cycle_out_channel * cycle_batch) * w_prec
        return total_cycle