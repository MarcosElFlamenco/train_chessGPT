import evaluation.gen_and_val_noS as a


INPUT_PGN = "1.d4 e5 2.dxe5 d6 3.exd6 Bxd6 4.Nf3 Nf6 5.Nc3 O-O 6.a3 Nc6 7.e3 a6 8.Be2 h6 9.O-O Ne5 10.Bd2 Nxf3+ 11.Bxf3 Be5 12.Rc1 c6 13.Qe2 Qd6 14.Rfd1 Bxh2+ 15.Kh1 Be5 16.e4 Bxc3 17.Bxc3 Qe6 18.Rd3 Bd7 19.Rcd1 Rad8 20.Bxf6 gxf6 21.Rd6 Qe7 22.Rd1d2 Be6 23.Rxd8 Rxd8 24.Rxd8+ Qxd8 25.c4 Qd4 26.c5 Qxc5 27.Qd2 f5 28.exf5 Bxf5 29.Qxh6 Bg6 30.Be4 Bxe4 31.Qh4 Bg6 32.Qd8+ Kg7 33.Qc7 b5 34.b4 Qc1+ 35.Kh2 Qxa3 36.Qe5+ Kg8 37.Qe8+ Kg7 38.Qxc6 Qxb4 39.Qxa6 Qh4+ 40.Kg1 b4 41.Qa1+ Qf6 42.Qa4 Qc3 43.f3 b3 44.Qa3 Qc2 45.Kh2 b2"

print(a.parse_pgn(INPUT_PGN))