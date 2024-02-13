`ifndef __mux_9to1_V__
`define __mux_9to1_V__

module mux_9to1
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [7:0], 
    input  logic [3:0]             sel,    
    output logic [DATA_WIDTH-1:0]  out
);
    logic [DATA_WIDTH-1:0]  out_tmp;
    always_comb begin
        case (sel[2:0]) // synopsys infer_mux
            3'b000:  out_tmp = vec[0];
            3'b001:  out_tmp = vec[1];
            3'b010:  out_tmp = vec[2];
            3'b011:  out_tmp = vec[3];
            3'b100:  out_tmp = vec[4];
            3'b101:  out_tmp = vec[5];
            3'b110:  out_tmp = vec[6];
            3'b111:  out_tmp = vec[7];
            default: out_tmp = {DATA_WIDTH{1'bx}};
        endcase
    end

    always_comb begin
        if ( sel[3] == 1'b0 ) begin
            out = out_tmp;
        end else begin
            out = 0;
        end
    end
    
endmodule

`endif