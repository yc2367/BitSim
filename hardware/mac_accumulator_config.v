`ifndef __mac_accumulator_config_V__
`define __mac_accumulator_config_V__


module mac_accumulator_config
#(
    parameter DATA_WIDTH = 25,
	parameter VEC_LENGTH = 4
) (
	input  logic signed [DATA_WIDTH-1:0]  accum   [VEC_LENGTH-1:0],
	input  logic        [2:0]             sel,
	output logic signed [DATA_WIDTH+1:0]  result
);
	genvar j;
	localparam RESULT_WIDTH = DATA_WIDTH+2;
	logic signed [DATA_WIDTH:0]    accum_tmp_1  [VEC_LENGTH/2-1:0];
	logic signed [DATA_WIDTH+1:0]  accum_tmp_2;
	generate
		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			assign accum_tmp_1[j] = accum[2*j] + accum[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign accum_tmp_2 = accum_tmp_1[2*j] + accum_tmp_1[2*j+1];
		end
	endgenerate

    always_comb begin
        case (sel) // synopsys infer_mux
            3'b000:   result = 0;
            3'b001:   result = accum[0];
            3'b010:   result = accum_tmp_1[0];
            3'b011:   result = accum[1];
            3'b100:   result = accum_tmp_2;
            3'b101:   result = accum[2];
            3'b110:   result = accum_tmp_1[1];
            3'b111:   result = accum[3];
            default:  result = {RESULT_WIDTH{1'bx}};
        endcase
    end
endmodule

`endif