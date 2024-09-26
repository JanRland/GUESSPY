#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""
import numpy as np
from scipy.stats import beta, norm, cauchy, expon, gamma, invgamma, rayleigh
import pickle as pkl


class GUESSPy:
    
    def __init__(self, y, x_score, verbose=False, fitLoc=["center", "center"]):
        """
        

        Parameters
        ----------
        y : List
            List of true classes for the corresponding scores x_score.
        x_score : List
            List of predicted scores for corresponding classes y.
        verbose : Boolean
            If set to true analytic steps are printed.

        Returns
        -------
        None.

        """
        self.fitLoc=fitLoc
        self.verbose=verbose
        self.y=np.array(y)
        self.x_score=np.array(x_score)
        
        self.c_0, self.c_1 = self.getClasses()
        N_c0=self.c_0.shape[0]
        N_c1=self.c_1.shape[0]
        N=float(N_c0+N_c1)
        
        self.p_c0=N_c0/N
        self.p_c1=N_c1/N
        
        self.implementedFitFunctions=["Normal", "Beta", "Cauchy", "Expon", \
                                      "Gamma", "InvGamma", "Rayleigh"]
        self.llhood_pdf_c0=None
        self.llhood_pdf_c1=None
        self.cal_c0=None
        self.cal_c1=None
        
        if self.verbose:
            self.resultBreak()
            print("Implemented fit functions")
            for f in self.implementedFitFunctions:
                print(f)
            self.resultBreak()
            
            self.resultBreak()
            print("Class Probabilities:")
            print("Class 0: " + str(self.p_c0))
            print("Class 1: " + str(self.p_c1))
            self.resultBreak()
            
    def getBins(self, N, x):
        """
        

        Parameters
        ----------
        N : Int
            Number of bins
        x : Numpy Array
            Data that should be binned.

        Returns
        -------
        binWidth : Float
            Width of the bins.
        binCenters : List
            List of bin centers (float).
        binWeights : List
            List of bin weights (int).
        N_total : Int
            Number of data points. 

        """
        binWidth=1./N
        binCenters=[]
        binWeights=[]
        binMeans=[]
        
        for i in range(N):
            binCenters.append(binWidth/2.+i*binWidth)
            if i==N-1:
                con=np.where((x>=i*binWidth) & (x<=(i+1)*binWidth))
            else:
                con=np.where((x>=i*binWidth) & (x<(i+1)*binWidth))
            nElements=len(x[con].tolist())
            binWeights.append(nElements)
            if nElements==0:
                binMeans.append(0.)
            else:
                binMeans.append(np.mean(x[con]))
        N_total=np.sum(binWeights)
        
        return binWidth, binCenters, binWeights, N_total, binMeans
    
    def getEmptyBins(self, N):
        """
        

        Parameters
        ----------
        N : Int
            Number of bins

        Returns
        -------
        binWidth : Float
            Width of the bins.
        binCenters : List
            List of bin centers (float).

        """
        binWidth=1./N
        binCenters=[]
        
        for i in range(N):
            binCenters.append(binWidth/2.+i*binWidth)
        
        return binWidth, binCenters
        
        
    def getBestLikelihood(self, c, N):
        """
        

        Parameters
        ----------
        c : Int
            Class for which the best fit function is searched for. Valid values
            are 0 and 1
        N : Int
            Number of bins for the data.

        """
        if self.verbose:
            self.resultBreak()
            print("Get likelihood for class " + str(c))
        if c==0:
            binWidth, binCenters, binWeights, N_total, _=self.getBins(N, self.c_0)
            
            if self.fitLoc[0]=="left":
                binCenters=list(np.array(binCenters)-binWidth/2.)
            elif self.fitLoc[0]=="right":
                binCenters=list(np.array(binCenters)+binWidth/2.)
            else:
                pass
            
            #
            # Fit Distributions
            #
            data=[]
            for i, bc in enumerate(binCenters):
                data+=[bc]*binWeights[i]
            params, llhoods = self.fitFunctions(data)
            
            #
            # Print Results
            #
            if self.verbose:
                self.resultBreak()
                print("Log-likelihood Values:")
                for i,p in enumerate(params):
                    print(self.implementedFitFunctions[i] + ": " + str(llhoods[i]) + " ----- " + str(p))
                self.resultBreak()
            
            print(llhoods)
            bestFit=np.nanmax(llhoods)
            print(bestFit)
            bestFit_index=llhoods.index(bestFit)
            self.llhood_pdf_c0=[self.implementedFitFunctions[bestFit_index], params[bestFit_index], llhoods[bestFit_index]]

            if self.verbose:
                self.resultBreak()
                print("Best Fit:")
                print("PDF: " + str(self.llhood_pdf_c0[0]))
                print("Params: " + str(self.llhood_pdf_c0[1]))
                print("Loglikelihood: " + str(self.llhood_pdf_c0[2]))
                self.resultBreak()
            
            
        elif c==1:
            binWidth, binCenters, binWeights, N_total, _=self.getBins(N, self.c_1)
            
            if self.fitLoc[1]=="left":
                binCenters=list(np.array(binCenters)-binWidth/2.)
            elif self.fitLoc[1]=="right":
                binCenters=list(np.array(binCenters)+binWidth/2.)
            else:
                pass
            
            #
            # Fit Distributions
            #
            data=[]
            for i, bc in enumerate(binCenters):
                data+=[bc]*binWeights[i]
            params, llhoods = self.fitFunctions(data)
            
            #
            # Print Results
            #
            if self.verbose:
                self.resultBreak()
                print("Log-likelihood Values:")
                for i,p in enumerate(params):
                    print(self.implementedFitFunctions[i] + ": " + str(llhoods[i]) + " ----- " + str(p))
                self.resultBreak()
            
            bestFit=np.amax(llhoods)
            bestFit_index=llhoods.index(bestFit)
            self.llhood_pdf_c1=[self.implementedFitFunctions[bestFit_index], params[bestFit_index], llhoods[bestFit_index]]
            if self.verbose:
                self.resultBreak()
                print("Best Fit:")
                print("PDF: " + str(self.llhood_pdf_c1[0]))
                print("Params: " + str(self.llhood_pdf_c1[1]))
                print("Loglikelihood: " + str(self.llhood_pdf_c1[2]))
                self.resultBreak()
            
        else:
            print("Invalid Class")
            print("Either 0 or 1 is implemented")
        if self.verbose:
            self.resultBreak()
    
        
    def getCalibration(self, x_i, N, epsilon=0.0001):
        """
        

        Parameters
        ----------
        x_i : Score Value
            Prediction that should be calibrated
        N : Int
            Number of bins for the data.
        epsilon : Float, optional
            Resolution for the probability calculation

        Returns
        -------
        p_cal_c0 : Float
            Probability of class 1 given the score x_i
        p_cal_c1 : Float
            Probability of class 1 given the score x_i

        """
        if self.llhood_pdf_c0 is None:
            self.getBestLikelihood(0, N)
            
        if self.llhood_pdf_c1 is None:
            self.getBestLikelihood(1, N)
        
        if self.verbose:
            self.resultBreak()
            print("Determining the probabilities for x_i="+ str(x_i))
            print("P(y=0)=" + str(self.p_c0))
            print("P(y=1)" + str(self.p_c1))
    
        p_y_0=self.getLikelihoodCDF(x_i+epsilon, 0)-self.getLikelihoodCDF(x_i-epsilon, 0)
        if self.verbose:
            print("P(x_i=" + str(x_i) +"|y=0)=" + str(p_y_0))
            
        p_y_1=self.getLikelihoodCDF(x_i+epsilon, 1)-self.getLikelihoodCDF(x_i-epsilon, 1)
        if self.verbose:
            print("P(x_i=" + str(x_i) +"|y=1)=" + str(p_y_1))
        
        p_x_i=p_y_0*self.p_c0+p_y_1*self.p_c1
        if self.verbose:
            print("P(x_i=" + str(x_i) +")=" + str(p_x_i))
        
        p_cal_c0=p_y_0*self.p_c0/p_x_i
        p_cal_c1=p_y_1*self.p_c1/p_x_i
        
        if self.verbose:
            self.resultBreak()
            print('P(y=0|x_i=' + str(x_i) + ')=' + str(p_cal_c0))
            print('P(y=1|x_i=' + str(x_i) + ')=' + str(p_cal_c1))
            self.resultBreak()
        
        return p_cal_c0, p_cal_c1

    def getLikelihood(self, x, c):
        """

        Parameters
        ----------
        x : Float
            Score Value for which the likelihood should be calculated. 
        c : Int
            Class for which the best fit function is searched for. Valid values
            are 0 and 1

        Returns
        -------
        res : Float
            Probabiliy of score x given class x.

        """
        res=None
        if c==0:
            if self.llhood_pdf_c0 is None:
                print("No likelihood was fitted yet")
            else:
                if self.llhood_pdf_c0[0]=="Normal":
                    res=norm.pdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="Beta":
                    res=beta.pdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="Cauchy":
                    res=cauchy.pdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="Expon":
                    res=expon.pdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="Gamma":
                    res=gamma.pdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="InvGamma":
                    res=invgamma.pdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="Rayleigh":
                    res=rayleigh.pdf(x,*self.llhood_pdf_c0[1])
                else:
                    print("Unexpected Error has occurred")
        elif c==1:
            if self.llhood_pdf_c1 is None:
                print("No likelihood was fitted yet")
            else:
                if self.llhood_pdf_c1[0]=="Normal":
                    res=norm.pdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="Beta":
                    res=beta.pdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="Cauchy":
                    res=cauchy.pdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="Expon":
                    res=expon.pdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="Gamma":
                    res=gamma.pdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="InvGamma":
                    res=invgamma.pdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="Rayleigh":
                    res=rayleigh.pdf(x,*self.llhood_pdf_c1[1])
                else:
                    print("Unexpected Error has occurred")
        else:
            print("Invalid Class")
            print("Either 0 or 1 is implemented")
        return res

    def getLikelihoodCDF(self, x, c):
        """
        

        Parameters
        ----------
        x : Float
            Value for which the CDF should be calculated
        c : Int
            Class

        Returns
        -------
        res : Float
            CDF of the pdf fitted on the score data.

        """
        
        res=None
        if c==0:
            if self.llhood_pdf_c0 is None:
                print("No likelihood was fitted yet")
            else:
                if self.llhood_pdf_c0[0]=="Normal":
                    res=norm.cdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="Beta":
                    res=beta.cdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="Cauchy":
                    res=cauchy.cdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="Expon":
                    res=expon.cdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="Gamma":
                    res=gamma.cdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="InvGamma":
                    res=invgamma.cdf(x,*self.llhood_pdf_c0[1])
                elif self.llhood_pdf_c0[0]=="Rayleigh":
                    res=rayleigh.cdf(x,*self.llhood_pdf_c0[1])
                else:
                    print("Unexpected Error has occurred")
        elif c==1:
            if self.llhood_pdf_c1 is None:
                print("No likelihood was fitted yet")
            else:
                if self.llhood_pdf_c1[0]=="Normal":
                    res=norm.cdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="Beta":
                    res=beta.cdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="Cauchy":
                    res=cauchy.cdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="Expon":
                    res=expon.cdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="Gamma":
                    res=gamma.cdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="InvGamma":
                    res=invgamma.cdf(x,*self.llhood_pdf_c1[1])
                elif self.llhood_pdf_c1[0]=="Rayleigh":
                    res=rayleigh.cdf(x,*self.llhood_pdf_c1[1])
                else:
                    print("Unexpected Error has occurred")
        else:
            print("Invalid Class")
            print("Either 0 or 1 is implemented")
        return res
    
    def getClasses(self):
        c_0=self.x_score[np.where(self.y==0)]
        c_1=self.x_score[np.where(self.y==1)]
        return c_0, c_1
    
    def fitFunctions(self, x):
        """
        

        Parameters
        ----------
        x : Numpy Array
            Array of score values given only one class (c=0 or c=1)

        Returns
        -------
        parameters : List
            Parameters for the PDFs.
        logLikelihoods : List
            List of log-likelihood values

        """
        loc_p=np.mean(x)
        scale_p=np.std(x)
        
        parameters=[]
        logLikelihoods=[]
        llhood=0
        
        #
        # Fit Norm function 
        #
        loc_norm, scale_norm = norm.fit(x, loc=loc_p, scale=scale_p)
        parameters.append([loc_norm, scale_norm])
        llhood=np.sum(np.log(norm.pdf(x,loc_norm,scale_norm)))
        logLikelihoods.append(llhood)
        
        #
        # Fit Beta
        #
        a_beta, b_beta, loc_beta, scale_beta = beta.fit(x, loc=loc_p, scale=scale_p)
        parameters.append([a_beta, b_beta, loc_beta, scale_beta])
        llhood=np.sum(np.log(beta.pdf(x, a_beta, b_beta, loc_beta, scale_beta )))
        logLikelihoods.append(llhood)
        
        #
        # Fit Cauchy
        #
        loc_cauchy, scale_cauchy = cauchy.fit(x, loc=loc_p, scale=scale_p)
        parameters.append([loc_cauchy, scale_cauchy])
        llhood=np.sum(np.log(cauchy.pdf(x, loc_cauchy, scale_cauchy)))
        logLikelihoods.append(llhood)
        
        #
        # Fit Expon
        #
        loc_expon, scale_expon = expon.fit(x, loc=loc_p, scale=scale_p)
        parameters.append([loc_expon, scale_expon])
        llhood=np.sum(np.log(expon.pdf(x, loc_expon, scale_expon)))
        logLikelihoods.append(llhood)
        
        #
        # Fit Gamma
        #
        a_gamma, loc_gamma, scale_gamma = gamma.fit(x, loc=loc_p, scale=scale_p)
        parameters.append([a_gamma, loc_gamma, scale_gamma])
        llhood=np.sum(np.log(gamma.pdf(x, a_gamma, loc_gamma, scale_gamma)))
        logLikelihoods.append(llhood)
        
        #
        # Fit InvGamma
        #
        a_invgamma, loc_invgamma, scale_invgamma = invgamma.fit(x, loc=loc_p, scale=scale_p)
        parameters.append([a_invgamma, loc_invgamma, scale_invgamma])
        llhood=np.sum(np.log(invgamma.pdf(x, a_invgamma, loc_invgamma, scale_invgamma)))
        logLikelihoods.append(llhood)
        
        #
        # Fit Rayleigh
        #
        loc_rayleigh, scale_rayleigh = rayleigh.fit(x, loc=loc_p, scale=scale_p)
        parameters.append([loc_rayleigh, scale_rayleigh])
        llhood=np.sum(np.log(rayleigh.pdf(x, loc_rayleigh, scale_rayleigh)))
        logLikelihoods.append(llhood)
        
        return parameters, logLikelihoods
    
    def saveState(self, f):
        """
        

        Parameters
        ----------
        f : String
            Name of the file the results should be saved to. Do not forget
            the file extension ".pkl" as it will be saved as pkl. 

        Returns
        -------
        None.

        """
        with open(f, 'wb') as f_pkl:
            pkl.dump([self.p_c0, self.p_c1, self.llhood_pdf_c0, self.llhood_pdf_c1], f_pkl)
    
    def loadState(self, f):
        """
        

        Parameters
        ----------
        f : String
            Name of the file containing the information.

        Returns
        -------
        None.

        """
        with open(f, 'rb') as f_pkl:
            self.p_c0, self.p_c1, self.llhood_pdf_c0, self.llhood_pdf_c1=pkl.load(f_pkl)
        N_c0=self.c_0.shape[0]
        N_c1=self.c_1.shape[0]
        N=float(N_c0+N_c1)
        
        self.p_c0=N_c0/N
        self.p_c1=N_c1/N
        
    def getCalibratedScores(self, c, N, epsilon=0.0001):
        if c==0:
            self.cal_c0=[]
            for score in self.c_0:
                cal_c0, _ =self.getCalibration(score, N, epsilon)
                self.cal_c0.append(cal_c0)
            self.cal_c0=np.array(self.cal_c0)
        elif c==1:
            self.cal_c1=[]
            for score in self.c_1:
                _, cal_c1 =self.getCalibration(score, N, epsilon)
                self.cal_c1.append(cal_c1)
            self.cal_c1=np.array(self.cal_c1)
            
        else:
            print("Unexpected Error has occurred")
                
            
    def ECE(self, c, N, useClibrated=False):
        res=0
        if useClibrated:

            binWidth, binCenters = self.getEmptyBins(N)
            self.getCalibratedScores(0, N, epsilon=binWidth/2.)
            self.getCalibratedScores(1, N, epsilon=binWidth/2.)
            
            binWidth_c0, binCenters_c0, binWeights_c0, N_total_c0, binMeans_c0=self.getBins(N, self.cal_c0)
            binWidth_c1, binCenters_c1, binWeights_c1, N_total_c1, binMeans_c1=self.getBins(N, self.cal_c1)
            N_total=N_total_c0+N_total_c1
            
        else:
            binWidth_c0, binCenters_c0, binWeights_c0, N_total_c0, binMeans_c0=self.getBins(N, self.c_0)
            binWidth_c1, binCenters_c1, binWeights_c1, N_total_c1, binMeans_c1=self.getBins(N, self.c_1)
            N_total=N_total_c0+N_total_c1
            
        if c==0:
            for i,bc in enumerate(binCenters_c0):
                p_i=(binWeights_c0[i]+binWeights_c1[i])/float(N_total)
                o_i=binWeights_c0[i]/float(N_total_c0)
                e_i=binMeans_c0[i]
                res+=p_i*np.abs(o_i-e_i)
        elif c==1:
            for i,bc in enumerate(binCenters_c1):
                p_i=(binWeights_c0[i]+binWeights_c1[i])/float(N_total)
                o_i=binWeights_c1[i]/float(N_total_c1)
                e_i=binMeans_c1[i]
                res+=p_i*np.abs(o_i-e_i)
        return res
        
    def MCE(self, c, N,  useClibrated=False):
        res=[]
        if useClibrated:

            binWidth, binCenters = self.getEmptyBins(N)
            self.getCalibratedScores(0, N, epsilon=binWidth/2.)
            self.getCalibratedScores(1, N, epsilon=binWidth/2.)
            
            binWidth_c0, binCenters_c0, binWeights_c0, N_total_c0, binMeans_c0=self.getBins(N, self.cal_c0)
            binWidth_c1, binCenters_c1, binWeights_c1, N_total_c1, binMeans_c1=self.getBins(N, self.cal_c1)
            
        else:
            binWidth_c0, binCenters_c0, binWeights_c0, N_total_c0, binMeans_c0=self.getBins(N, self.c_0)
            binWidth_c1, binCenters_c1, binWeights_c1, N_total_c1, binMeans_c1=self.getBins(N, self.c_1)
            
        if c==0:
            for i,bc in enumerate(binCenters_c0):
                o_i=binWeights_c0[i]/float(N_total_c0)
                e_i=binMeans_c0[i]
                res.append(np.abs(o_i-e_i))
        elif c==1:
            for i,bc in enumerate(binCenters_c1):
                o_i=binWeights_c1[i]/float(N_total_c1)
                e_i=binMeans_c1[i]
                res.append(np.abs(o_i-e_i))

        res=np.amax(res)
        return res
    
    def LCE(self, c, N, useClibrated=False):
        res=0
        if useClibrated:

            binWidth, binCenters = self.getEmptyBins(N)
            self.getCalibratedScores(0, N, epsilon=binWidth/2.)
            self.getCalibratedScores(1, N, epsilon=binWidth/2.)
            
            binWidth_c0, binCenters_c0, binWeights_c0, N_total_c0, binMeans_c0=self.getBins(N, self.cal_c0)
            binWidth_c1, binCenters_c1, binWeights_c1, N_total_c1, binMeans_c1=self.getBins(N, self.cal_c1)
            
        else:
            binWidth_c0, binCenters_c0, binWeights_c0, N_total_c0, binMeans_c0=self.getBins(N, self.c_0)
            binWidth_c1, binCenters_c1, binWeights_c1, N_total_c1, binMeans_c1=self.getBins(N, self.c_1)
            
        if c==0:
            for i,bc in enumerate(binCenters_c0):
                o_i=binWeights_c0[i]/float(N_total_c0)
                e_i=binMeans_c0[i]
                res+=o_i*np.abs(e_i)
        elif c==1:
            for i,bc in enumerate(binCenters_c1):
                o_i=binWeights_c1[i]/float(N_total_c1)
                e_i=binMeans_c1[i]
                res+=o_i*np.abs(1-e_i)
        return res
    
    def resultBreak(self):
        print("###################################")
