import zmq 
import cv2 
import lib.face_blending




def main():
    context = zmq.sugar.Context()
    sock = context.socket(zmq.REP)     
    sock.bind("tcp://*:5555")

    return 


if __name__ == "__main__":

    main()
