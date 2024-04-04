`ifndef __mac_unit_Pragmatic_8_V__
`define __mac_unit_Pragmatic_8_V__


module pos_neg_select #(
	parameter DATA_WIDTH = 8
) (
	input  logic signed [DATA_WIDTH-1:0] in,
	input  logic                         sign,	
	output logic signed [DATA_WIDTH-1:0] out
); 
	always_comb begin
		if (sign) begin
			out = ~in + 1'b1;
		end else begin
			out = in;
		end
	end
endmodule


module shifter_1st_stage #(
	parameter IN_WIDTH  = 8,
	parameter OUT_WIDTH = IN_WIDTH + 3
) (
	input  logic signed [IN_WIDTH-1:0]  in,
	input  logic        [1:0]           shift_sel,
	input  logic                        en, 	
	output logic signed [OUT_WIDTH-1:0] out
);
	logic signed [OUT_WIDTH-1:0] out_tmp;
	always_comb begin 
		case (shift_sel)
			2'b00 :  out_tmp = in;
			2'b01 :  out_tmp = in <<< 1;
			2'b10 :  out_tmp = in <<< 2;
			2'b11 :  out_tmp = in <<< 3;
			default: out_tmp = {OUT_WIDTH{1'bx}};
		endcase
	end

	always_comb begin
		if (en) begin
			out = out_tmp;
		end else begin
			out = 0;
		end
	end
endmodule


module shifter_2nd_stage #(
	parameter IN_WIDTH  = 8,
	parameter OUT_WIDTH = IN_WIDTH + 7
) (
	input  logic signed [IN_WIDTH-1:0]  in,
	input  logic        [2:0]           shift_sel,
	input  logic                        en, 	
	output logic signed [OUT_WIDTH-1:0] out
);
	logic signed [OUT_WIDTH-1:0] out_tmp;
	always_comb begin 
		case (shift_sel)
			3'b000 :  out_tmp = in <<< 0;
			3'b001 :  out_tmp = in <<< 1;
			3'b010 :  out_tmp = in <<< 2;
			3'b011 :  out_tmp = in <<< 3;
			3'b100 :  out_tmp = in <<< 4;
			3'b101 :  out_tmp = in <<< 5;
			3'b110 :  out_tmp = in <<< 6;
			3'b111 :  out_tmp = in <<< 7;
			default:  out_tmp = {OUT_WIDTH{1'bx}};
		endcase
	end

	always_comb begin
		if (en) begin
			out = out_tmp;
		end else begin
			out = 0;
		end
	end
endmodule


module mac_unit_Pragmatic_8
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act_in         [VEC_LENGTH-1:0], 
	input  logic        [1:0]                shift_1st_sel  [VEC_LENGTH-1:0],
	input  logic                             shift_1st_en   [VEC_LENGTH-1:0],
	input  logic        [2:0]                shift_2nd_sel, // 2nd-stage shifter
	input  logic                             shift_2nd_en,  // whether enable 2nd-stage shifter
	input  logic                             is_neg         [VEC_LENGTH-1:0],

	input  logic signed [ACC_WIDTH-1:0]      accum_prev,
	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;

	logic signed [DATA_WIDTH-1:0]  act_tmp    [VEC_LENGTH-1:0] ;
	logic signed [DATA_WIDTH+2:0]  act_out    [VEC_LENGTH-1:0] ;
	generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
			pos_neg_select #(DATA_WIDTH) pos_neg_act (
				.in(act_in[j]), .out(act_tmp[j]), .sign(is_neg[j])
			);
			shifter_1st_stage #(DATA_WIDTH, DATA_WIDTH+3) shift_1st (
				.in(act_tmp[j]), .shift_sel(shift_1st_sel[j]), .en(shift_1st_en[j]), .out(act_out[j])
			);
		end
	endgenerate

	logic signed [DATA_WIDTH+3:0]  psum_1 [VEC_LENGTH/2-1:0];
	logic signed [DATA_WIDTH+4:0]  psum_2 [VEC_LENGTH/4-1:0];
	logic signed [DATA_WIDTH+4:0]  psum_3;
	generate
		for (j=0; j<VEC_LENGTH/2; j=j+1) begin
			assign psum_1[j] = act_out[2*j] + act_out[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/4; j=j+1) begin
			assign psum_2[j] = psum_1[2*j] + psum_1[2*j+1];
		end

		for (j=0; j<VEC_LENGTH/8; j=j+1) begin
			assign psum_3 = psum_2[2*j] + psum_2[2*j+1];
		end
	endgenerate

	logic signed [DATA_WIDTH+11:0]  psum_total;
	shifter_2nd_stage #(DATA_WIDTH+5, DATA_WIDTH+12) shift_2nd (
		.in(psum_3), .shift_sel(shift_2nd_sel), .en(shift_2nd_en), .out(psum_total)
	);

	logic signed [ACC_WIDTH-1:0]  accum_in, accum_out;
	always_comb begin
		if (load_accum) begin
			accum_in = accum_prev;
		end else begin
			accum_in = accum_out;
		end
	end

	logic signed [DATA_WIDTH+11:0]  psum_total_reg;
	always @(posedge clk) begin
		if (reset) begin
			accum_out <= 0;
			psum_total_reg <= 0;
		end else if	(en) begin
			psum_total_reg <= psum_total;
			accum_out <= psum_total_reg + accum_in;
		end
	end

	assign result = accum_out[ACC_WIDTH-1:ACC_WIDTH-16];

endmodule


module mac_unit_Pragmatic_8_clk
#(
    parameter DATA_WIDTH    = 8,
	parameter VEC_LENGTH    = 8,
	parameter ACC_WIDTH     = DATA_WIDTH + 16,
	parameter RESULT_WIDTH  = 2*DATA_WIDTH
) (
	input  logic                             clk,
	input  logic                             reset,
	input  logic                             en,
	input  logic                             load_accum,

	input  logic signed [DATA_WIDTH-1:0]     act               [VEC_LENGTH-1:0], 
	input  logic        [1:0]                shift_1st_sel_in  [VEC_LENGTH-1:0],
	input  logic                             shift_1st_en_in   [VEC_LENGTH-1:0],
	input  logic        [2:0]                shift_2nd_sel_in, // 2nd-stage shifter
	input  logic                             shift_2nd_en_in,  // whether enable 2nd-stage shifter
	input  logic                             is_neg_in         [VEC_LENGTH-1:0],

	input  logic signed [ACC_WIDTH-1:0]      accum_prev,
	output logic signed [RESULT_WIDTH-1:0]   result
);
	genvar j;
	
	logic signed [DATA_WIDTH-1:0]  act_in [VEC_LENGTH-1:0];

	logic        [1:0]                shift_1st_sel  [VEC_LENGTH-1:0];
	logic                             shift_1st_en   [VEC_LENGTH-1:0];
	logic        [2:0]                shift_2nd_sel; // 2nd-stage shifter
	logic                             shift_2nd_en;  // whether enable 2nd-stage shifter
	logic                             is_neg         [VEC_LENGTH-1:0];
	generate
		for (j=0; j<VEC_LENGTH; j=j+1) begin
			always @(posedge clk) begin
				if (reset) begin
					act_in[j] <= 0;
					shift_1st_sel[j] <= 0;
					shift_1st_en[j] <= 0;
					is_neg[j] <= 0;
				end else begin
					act_in[j] <= act[j];
					shift_1st_sel[j] <= shift_1st_sel_in[j];
					shift_1st_en[j] <= shift_1st_en_in[j];
					is_neg[j] <= is_neg_in[j];
				end
			end
		end
	endgenerate

	always @(posedge clk) begin
		if (reset) begin
			shift_2nd_sel <= 0;
			shift_2nd_en <= 0;
		end else begin
			shift_2nd_sel <= shift_2nd_sel_in;
			shift_2nd_en <= shift_2nd_en_in;
		end
	end

	mac_unit_Pragmatic_8 #(DATA_WIDTH, VEC_LENGTH, ACC_WIDTH, RESULT_WIDTH) mac (.*);
endmodule

`endif