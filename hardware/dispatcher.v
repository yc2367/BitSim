`ifndef __dispatcher_V__
`define __dispatcher_V__

module dispatcher
#(
    parameter DATA_WIDTH = 128
) (
    input  logic [DATA_WIDTH-1:0]  vec [15:0], 
    input  logic [3:0]             sel,     
    output logic [DATA_WIDTH-1:0]  out
);
    always_comb begin
        case (sel) // synopsys infer_mux
            4'b0000: out = vec[0];
            4'b0001: out = vec[1];
            4'b0010: out = vec[2];
            4'b0011: out = vec[3];
            4'b0100: out = vec[4];
            4'b0101: out = vec[5];
            4'b0110: out = vec[6];
            4'b0111: out = vec[7];
            4'b1000: out = vec[8];
            4'b1001: out = vec[9];
            4'b1010: out = vec[10];
            4'b1011: out = vec[11];
            4'b1100: out = vec[12];
            4'b1101: out = vec[13];
            4'b1110: out = vec[14];
            4'b1111: out = vec[15];
            default:  out = {DATA_WIDTH{1'bx}};
        endcase
    end
endmodule

module dispatcher_clk
#(
    parameter DATA_WIDTH = 128
) (
    input  logic                   clk,
    input  logic                   reset,

    input  logic [DATA_WIDTH-1:0]  vec     [15:0], 
    input  logic [3:0]             sel_reg,     
    output logic [DATA_WIDTH-1:0]  out_reg
);
    logic [3:0]             sel;
    logic [DATA_WIDTH-1:0]  out;

    dispatcher #(DATA_WIDTH) m (.*);

    always @(posedge clk) begin
		if (reset) begin
			out_reg <= 0;
		end else begin
            sel     <= sel_reg;
            out_reg <= out;
		end
	end
endmodule

`endif