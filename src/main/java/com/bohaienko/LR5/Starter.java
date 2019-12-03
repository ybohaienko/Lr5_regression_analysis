package com.bohaienko.LR5;

import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;

@Service
public class Starter {

	private LinearRegression linearRegressionModel;

	@EventListener(ApplicationReadyEvent.class)
	public void init() throws Exception {
		String file = "src/main/resources/winequality-red.arff";
		Instances instances = new Instances(new BufferedReader(new FileReader(file)));

		setLinearRegressionModel(instances);
		classifyInstance(instances.lastInstance());
	}

	private void setLinearRegressionModel(Instances instances) throws Exception {
		instances.setClassIndex(instances.numAttributes() - 1);
		linearRegressionModel = new LinearRegression();
		linearRegressionModel.buildClassifier(instances);
		System.out.println(linearRegressionModel);
	}

	private void classifyInstance(Instance instance) throws Exception {
		double quality = linearRegressionModel.classifyInstance(instance);
		System.out.println("Test wine instance [" + instance + "] quality: " + quality);
	}
}
