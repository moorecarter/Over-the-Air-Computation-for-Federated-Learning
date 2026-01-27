import numpy as np
import argparse


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', help="Root for Zadoff Chu sequence generation (u).", required=True, type=int)
    parser.add_argument('-N' ,help="Length of Zadoff Chu sequence to be generated (N).", required=True, type=int)
    parser.add_argument('-q' ,help="Cyclic shift of Zadoff Chu sequence to be generated (q) [DEFAULT=0].", required=False, type=int, default=0)
    return parser.parse_args()

def gcd_euclidean(a, b):
    if a == 0:
        return b
    return gcd_euclidean(b % a, a)

if __name__ == "__main__":
    try:
        args = parseArgs()
        if not(args.u):
            raise Exception("Please specify a root (-u <ROOT>)")
        if not(args.N):
            raise Exception("Please specify a length (-N <LENGTH>)")
        if args.u < 0 or args.u >= args.N:
            raise Exception("Please choose u between 0 and N.")
        if args.N < 0:
            raise Exception("Please choose N > 0.")
        if(gcd_euclidean(args.N, args.u) != 1):
            raise Exception("Please choose N and u such that they are coprime.")
        
        n = np.arange(args.N)
        cf = args.N % 2
        arg = -1j * (np.pi * args.u * n * (n + cf + 2 * args.q)) / args.N
        x = np.exp(arg).astype(np.complex64)
        print(f"Zadoff Chu Sequence: {x}")
        x.tofile("zc_seq.bin")
        exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(-1)
    