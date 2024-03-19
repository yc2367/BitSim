`ifndef __mux_5to1_V__
`define __mux_5to1_V__

module mux_5to1
#(
    parameter DATA_WIDTH = 8
) (
    input  logic [DATA_WIDTH-1:0]  vec [4:0], 
    input  logic [2:0]             sel,  
    input  logic                   val,         
    output logic [DATA_WIDTH-1:0]  out
);

    logic [DATA_WIDTH-1:0]  out_tmp;
    always_comb begin
        case (sel) 
            3'b000:   out_tmp = vec[0];
            3'b001:   out_tmp = vec[1];
            3'b010:   out_tmp = vec[2];
            3'b011:   out_tmp = vec[3];
            3'b100:   out_tmp = vec[4];
            default:  out_tmp = {DATA_WIDTH{1'bx}};
        endcase
    end

    always_comb begin
        if ( val ) begin
            out = out_tmp;
        end else begin
            out = 0;
        end
    end

endmodule

`endif