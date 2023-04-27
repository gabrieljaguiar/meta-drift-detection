"""
public class BinaryImbalancedGenerator extends AbstractOptionHandler implements InstanceStream {
	
	private static final long serialVersionUID = 1L;

	@Override
	public String getPurposeString() {
		return "Generates a static imbalance ratio distribution from a provided stream generator.";
	}
	
	public IntOption seedOption = new IntOption("seed", 's',
			"Seed for random generation of classes distribution.", 1);
	
	public FloatOption imbalanceRatioOption = new FloatOption("imbalanceRatio", 'i',
			"Percentage of minority class examples (default 10%, i.e. IR 10)", 0.1, 0, 1);
	
	public ClassOption generatorOption = new ClassOption("generator", 'g',
			"Class of the stream generator", InstanceStream.class, "moa.streams.generators.AgrawalGenerator");
	
	protected InstanceStream generator;
	
	protected Random instanceRandom;
	
	@Override
	public InstancesHeader getHeader() {
		return generator.getHeader();
	}

	@Override
	public long estimatedRemainingInstances() {
		return generator.estimatedRemainingInstances();
	}

	@Override
	public boolean hasMoreInstances() {
		return generator.hasMoreInstances();
	}

	@Override
	public Example<Instance> nextInstance() {
		
		int expectedClass = instanceRandom.nextFloat() < imbalanceRatioOption.getValue() ? 1 : 0;
		Example<Instance> instance = null;
		
		do {
			instance = generator.nextInstance();
		} while(instance.getData().classValue() != expectedClass);
		
		return instance; 
	}

	@Override
	public boolean isRestartable() {
		return generator.isRestartable();
	}

	@Override
	public void restart() {
		instanceRandom = new Random(seedOption.getValue());
		generator.restart();
	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		generator = (InstanceStream) getPreparedClassOption(this.generatorOption);
		restart();
	}
}


/*
 * Copyright (c) 2018.
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

package moa.streams.generators.imbalanced;

import java.util.Random;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.core.Example;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.ClassOption;
import moa.streams.ExampleStream;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;


/**
 * Imbalanced Stream.
 *
 * This is a meta-generator that produces class imbalance in a stream.
 * Only two parameters are required to be set:
 * - The original stream
 * - The ratio (proportion) of each class in the stream.
 *
 * The second parameter determines the ratio of each class in the output stream.
 * The ratio of each class should be provided as a real number between 0.0 and 1.0,
 * each being followed by a semicolon, and their sum should add up to 1.0.
 * The default value of 0.9;0.1 stands for an output stream where approximately 90%
 * of the instances belonging to the first class while the remainder 10% would
 * belong to the secondary class.
 *
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 * @version 1.0
 */

public class MultiClassImbalancedGenerator extends AbstractOptionHandler implements InstanceStream {

	private static final long serialVersionUID = 1L;

	public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to imbalance.", ExampleStream.class,
            "generators.RandomTreeGenerator");

    public StringOption classRatioOption = new StringOption("classRatio", 'c',
            "Determine the ratio of each class in the output stream. " +
                    "The ratio of each class should be given as a real number " +
                    "between 0 and 1, each followed by a semicolon, and their sum should be equal to 1. " +
                    "The default value of \"0.9;0.1\" stands for an output stream with approximately 90% " +
                    "of the instances belonging to the first class and 10% to the second class.",
            "0.9;0.1");

    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);

    protected ExampleStream originalStream    = null;
    protected double        probPerClass[]    = null;
    protected Random        random            = null;
    protected int           numClasses        = 0;

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        originalStream = (ExampleStream) getPreparedClassOption(streamOption);
        numClasses = originalStream.getHeader().numClasses();
        probPerClass = new double[numClasses];
        probPerClass = new double[numClasses];


        // checks and sets the probabilities
        double sumProbs = 0.0;
        String probs[] = classRatioOption.getValue().split(";");
        for(int i = 0; i < probs.length; i++){
            if(i > probPerClass.length - 1) throw new IllegalArgumentException("Please make sure the number of class " +
                    "ratios provided is less or equal the number of classes in the original stream.");
            Double p = Double.parseDouble(probs[i]);
            if(!p.isNaN() && ! p.isInfinite() && p.doubleValue() >= 0.0 && p.doubleValue() <= 1.0){
                // aka is a valid number between 0.0 and 1.0
                probPerClass[i]  = p;
                sumProbs        += p;
            }else{
                throw new IllegalArgumentException("Please make sure only numbers between 0.0 and 1.0 are inputted.");
            }
        }

        // checks if the values sum to 1.0
        if(Math.abs(sumProbs - 1.0) > 1e-3) throw new IllegalArgumentException("Please make sure the class ratios sum up to 1.0.");


        // initializes the random generator
        random = new Random(instanceRandomSeedOption.getValue());
    }

    @Override
    public InstancesHeader getHeader() {
        return originalStream.getHeader();
    }

    @Override
    public long estimatedRemainingInstances() {
        return originalStream.estimatedRemainingInstances();
    }

    @Override
    public boolean hasMoreInstances() {
        return originalStream.hasMoreInstances();
    }

    @Override
    public Example<Instance> nextInstance() {
        // a value between 0.0 and 1.0 uniformly distributed
        double p   = random.nextDouble();
        int iClass = -1;
        // loops over all class probabilities to see from which class the next instance should be from
        while(p > 0.0){
            iClass++;
            p -= probPerClass[iClass];
        }
        
        Example<Instance> instance = null;
        
        do {
			instance = originalStream.nextInstance();
		} while(instance.getData().classValue() != iClass);

        return instance;
    }

    @Override
    public boolean isRestartable() {
        return originalStream.isRestartable();
    }

    @Override
    public void restart() {
        this.random = new Random(instanceRandomSeedOption.getValue());
        this.originalStream.restart();
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {}

}


"""
