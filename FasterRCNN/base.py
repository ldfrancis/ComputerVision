import tensorflow as tf
import numpy as np

class Base(tf.keras.Model):
    def __init__(self, layer='block5_conv3'):
        super(Base, self).__init__()
        self.vgg16 = tf.keras.applications.VGG16()
        self.base_model = tf.keras.Model(inputs=[self.vgg16.input], outputs=[self.vgg16.get_layer(layer).output])

    def call(self, x):
        return self.base_model(x)
      
    def summary(self):
        return self.base_model.summary()


class RPN(tf.keras.Model):
    def __init__(self, k=9):
        super(RPN, self).__init__()
        self.k = k
        self.conv_base = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', name='conv_base')
        self.conv_reg = tf.keras.layers.Conv2D(4 *k, 1, padding='same', activation='relu', name='conv_reg')
        self.conv_cls = tf.keras.layers.Conv2D(2*k, 1, padding='same', activation='sigmoid', name='conv_cls')
        
    def call(self, x):
        conv_base = self.conv_base(x) # shape => bxHxWx512
        reg = self.conv_reg(conv_base) # shape => bxHxWx(4k)
        cl = self.conv_cls(conv_base) # shape => bxHxWx(2k)
        
        reg = tf.reshape(reg, list(reg.shape[:-1])+[self.k, 4])
        cl = tf.reshape(cl, list(cl.shape[:-1])+[self.k,2])
        
        self.reg = reg
        self.cls = cl
        
        return reg, cl
    
        
    def rpn_roi(self):
        pass
      
    def get_feature_map_anchors(self):
      
        _,H,W = self.reg.shape[:3]
        
        anchor_reg = np.zeros(self.reg.shape)
        anchor_cls = np.zeros(self.cls.shape)
        
        for h in H:
            for w in W:
                anchor_reg[:,h,w] = self.get_k_anchors(w,h)
                anchor_cls[:,h,w] = np.zeros((self.k,2))
                
        self.anchor_reg = tf.constant(anchor_reg) // 16
        self.anchor_cls = tf.constant(anchor_cls)


            
      
    def get_k_anchors(self,x,y):
        
        scales = np.array([[128,128],[256,256],[512,512]])
        aspects = np.array([[1,1],[1,2],[2,1]])
        k_anchors = np.zeros((9,4), np.float32)

        i = 0
        for s in scales:
            for a in aspects:
                k_anchors[i] = np.array([x,y] + list(s*a))
                i += 1
  
        return k_anchors


    def rpn_loss(self):
        a = 4
        