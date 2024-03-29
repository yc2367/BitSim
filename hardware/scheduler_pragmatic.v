`ifndef __scheduler_pragmatic_V__
`define __scheduler_pragmatic_V__

`include "reg_file.v"
`include "decoder_3to7.v"
`include "pencoder_7to3.v"
`include "mux_2to1.v"

module scheduler_pragmatic
#(
    parameter DATA_WIDTH = 8,
    parameter VEC_LENGTH = 8
) (
    input  logic                   clk,
    input  logic                   reset,
    input  logic                   en_comp, // compute read-enable
    input  logic                   wen_rf,  // register file write-enable

    input  logic [DATA_WIDTH-1:0]  weight        [VEC_LENGTH-1:0], // use sign-magnitude weight

    output logic [2:0]             oneffset      [VEC_LENGTH-1:0],
    output logic                   sign_oneffset [VEC_LENGTH-1:0],
    output logic                   val_oneffset  [VEC_LENGTH-1:0]
);
    genvar j;
    logic [DATA_WIDTH-1:0]  din_rf  [VEC_LENGTH-1:0];
    logic [DATA_WIDTH-1:0]  dout_rf [VEC_LENGTH-1:0];
    // during computation, the reg file output will be weight if read the next set of weights
    // otherwise, it will be the decoded current set of weights 
    logic [DATA_WIDTH-2:0]  decoded_weight [VEC_LENGTH-1:0]; 

    generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
            mux_2to1 #(DATA_WIDTH-1) mux (
                .in_1 (weight[j][DATA_WIDTH-2:0]), 
                .in_0 (decoded_weight[j]),
                .sel  (wen_rf),
                .out  (din_rf[j][DATA_WIDTH-2:0])
            );

            assign din_rf[j][DATA_WIDTH-1] = weight[j][DATA_WIDTH-1];
        end
    endgenerate 

    reg_file #(
        .DATA_WIDTH(DATA_WIDTH), .VEC_LENGTH(VEC_LENGTH)
    ) rf (
        .d_in(din_rf), .d_out(dout_rf), .w_en(wen_rf), .*
    );

    logic [2:0]             oneffset_tmp     [VEC_LENGTH-1:0];
    logic                   val_oneffset_tmp [VEC_LENGTH-1:0];
    logic [DATA_WIDTH-2:0]  dec_out [VEC_LENGTH-1:0];
    generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
            pencoder_7to3  pen (
                .bitmask(dout_rf[j][DATA_WIDTH-2:0]), 
                .out(oneffset_tmp[j]), .val(val_oneffset_tmp[j])
            );
            decoder_3to7   dec (
                .in(oneffset_tmp[j]), .val(val_oneffset_tmp[j]), .out(dec_out[j])
            );
            assign decoded_weight[j] = dout_rf[j][DATA_WIDTH-2:0] ^ dec_out[j];
        end
    endgenerate 
    
    generate
        for (j=0; j<VEC_LENGTH; j=j+1) begin
            always @(posedge clk) begin
                if (reset) begin
                    oneffset[j]      <= 0;
                    val_oneffset[j]  <= 0;
                    sign_oneffset[j] <= 0;
                end else if	(en_comp) begin
                    oneffset[j]      <= oneffset_tmp[j];
                    val_oneffset[j]  <= val_oneffset_tmp[j];
                    sign_oneffset[j] <= dout_rf[j][DATA_WIDTH-1];
                end
            end
        end
    endgenerate
endmodule

`endif