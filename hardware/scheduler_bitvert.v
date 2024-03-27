`ifndef __scheduler_bitvert_V__
`define __scheduler_bitvert_V__

`include "shift_reg_file.v"
`include "decoder_3to5.v"
`include "pencoder_5to3.v"
`include "mux_2to1.v"

module scheduler_bitvert
(
    input  logic        clk,
    input  logic        reset,
    input  logic        en_comp, // compute read-enable
    input  logic        ren_rf,  // register file read-enable
    input  logic        wen_rf,  // register file write-enable

    input  logic [7:0]  weight     [7:0],   

    output logic [2:0]  sel        [3:0],
    output logic        val        [3:0],
    output logic        skip_bit_1 
);
    localparam DATA_WIDTH = 8;
    localparam VEC_LENGTH = 8;

    logic [DATA_WIDTH-1:0]  bitmask, bitmask_inv;

    /*
    always @(posedge clk) begin
        if (reset) begin
            bitmask <= 0;
        end else if	(r_en) begin
            bitmask <= w_bit;
        end
    end
    */
    shift_reg_file #(
        .DATA_WIDTH(DATA_WIDTH), .VEC_LENGTH(VEC_LENGTH)
    ) rf (
        .d_in(weight), .d_out(bitmask), .r_en(ren_rf), .w_en(wen_rf), .*
    );
    
    assign bitmask_inv = ~bitmask;

    logic unsigned [1:0] sum_1 [DATA_WIDTH/2-1:0];
    logic unsigned [2:0] sum_2 [DATA_WIDTH/4-1:0];
    logic unsigned [3:0] sum_3 ;
    genvar j;
    generate
		for (j=0; j<DATA_WIDTH/2; j=j+1) begin
            assign sum_1[j] = bitmask[2*j] + bitmask[2*j+1];
        end
        for (j=0; j<DATA_WIDTH/4; j=j+1) begin
            assign sum_2[j] = sum_1[2*j] + sum_1[2*j+1];
        end
        for (j=0; j<DATA_WIDTH/8; j=j+1) begin
            assign sum_3[j] = sum_2[2*j] + sum_2[2*j+1];
        end
    endgenerate

    logic bitmask_sel;
    assign bitmask_sel = (sum_3 > (DATA_WIDTH/2));
    
    logic [DATA_WIDTH-1:0] bitmask_out;
    mux_2to1 #(DATA_WIDTH) mux (.in_0(bitmask), .in_1(bitmask_inv), .sel(bitmask_sel), .out(bitmask_out));

    logic [2:0]  act_idx      [3:0];
    logic        val_act      [3:0];
    logic [4:0]  pencoder_in  [3:0];
    logic [4:0]  dec_out      [3:1];
    assign pencoder_in[3] = bitmask_out[7:3];
    assign pencoder_in[2] = {pencoder_in[3][3:0] ^ dec_out[3][3:0], bitmask_out[2]};
    assign pencoder_in[1] = {pencoder_in[2][3:0] ^ dec_out[2][3:0], bitmask_out[1]};
    assign pencoder_in[0] = {pencoder_in[1][3:0] ^ dec_out[1][3:0], bitmask_out[0]};

    generate
		for (j=3; j>-1; j=j-1) begin
            pencoder_5to3  pen (.bitmask(pencoder_in[j]), .out(act_idx[j]), .val(val_act[j]));
        end
        for (j=3; j>0; j=j-1) begin
            decoder_3to5   dec (.in(act_idx[j]), .val(val_act[j]), .out(dec_out[j]));
        end

        for (j=0; j<4; j=j+1) begin
            always @(posedge clk) begin
                if (reset) begin
                    sel[j] <= 0;
                    val[j] <= 0;
                end else if	(en_comp) begin
                    sel[j] <= act_idx[j];
                    val[j] <= val_act[j];
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (reset) begin
            skip_bit_1 <= 0;
        end else if	(en_comp) begin
            skip_bit_1 <= bitmask_sel;
        end
    end
endmodule

`endif