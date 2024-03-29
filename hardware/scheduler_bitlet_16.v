`ifndef __scheduler_bitlet_16_V__
`define __scheduler_bitlet_16_V__

`include "reg_file.v"
`include "decoder_4to16.v"
`include "pencoder_16to4.v"
`include "mux_2to1.v"

module scheduler_bitlet_16
#(
    parameter DATA_WIDTH = 8,
    parameter VEC_LENGTH = 16,
    parameter MUX_SEL_WIDTH = $clog2(VEC_LENGTH)
) (
    input  logic                      clk,
    input  logic                      reset,
    input  logic                      en_comp, // compute read-enable
    input  logic                      wen_rf,  // register file write-enable

    input  logic [VEC_LENGTH-1:0]     weight        [DATA_WIDTH-1:0], // use sign-magnitude weight

    output logic [MUX_SEL_WIDTH-1:0]  act_sel       [DATA_WIDTH-1:0],
    output logic                      act_val       [DATA_WIDTH-1:0]
);
    genvar j;
    logic [VEC_LENGTH-1:0]  din_rf  [DATA_WIDTH-1:0];
    logic [VEC_LENGTH-1:0]  dout_rf [DATA_WIDTH-1:0];
    // during computation, the reg file output will be weight if read the next set of weights
    // otherwise, it will be the decoded current set of weights 
    logic [VEC_LENGTH-1:0]  decoded_weight [DATA_WIDTH-1:0]; 

    generate
		for (j=0; j<DATA_WIDTH; j=j+1) begin
            mux_2to1 #(VEC_LENGTH) mux (
                .in_1 (weight[j]), 
                .in_0 (decoded_weight[j]),
                .sel  (wen_rf),
                .out  (din_rf[j])
            );
        end
    endgenerate 

    reg_file #(
        .DATA_WIDTH(VEC_LENGTH), .VEC_LENGTH(DATA_WIDTH)
    ) rf (
        .d_in(din_rf), .d_out(dout_rf), .w_en(wen_rf), .*
    );

    logic [MUX_SEL_WIDTH-1:0] act_sel_tmp  [DATA_WIDTH-1:0];
    logic                     act_val_tmp  [DATA_WIDTH-1:0];
    logic [VEC_LENGTH-1:0]    dec_out      [DATA_WIDTH-1:0];
    generate
		for (j=0; j<DATA_WIDTH; j=j+1) begin
            pencoder_16to4  pen (
                .bitmask(dout_rf[j]), 
                .out(act_sel_tmp[j]), .val(act_val_tmp[j])
            );
            decoder_4to16   dec (
                .in(act_sel_tmp[j]), .val(act_val_tmp[j]), .out(dec_out[j])
            );
            assign decoded_weight[j] = dout_rf[j] ^ dec_out[j];
        end
    endgenerate 
    
    generate
        for (j=0; j<DATA_WIDTH; j=j+1) begin
            always @(posedge clk) begin
                if (reset) begin
                    act_sel[j]  <= 0;
                    act_val[j]  <= 0;
                end else if	(en_comp) begin
                    act_sel[j]  <= act_sel_tmp[j];
                    act_val[j]  <= act_val_tmp[j];
                end
            end
        end
    endgenerate
endmodule

`endif