"use client";

import React, { useEffect } from "react";
import { useGlucoTwinStore } from "../../lib/store";
import { fetchPatients, fetchPatientCGM } from "../../lib/api";
import { PatientSelector } from "../../components/PatientSelector";
import { WhatIfSimulator } from "../../components/WhatIfSimulator";
import { PatientProfile } from "../../lib/types";

export default function SimulatorStandalonePage() {
  const {
    patients,
    selectedPatient,
    setPatients,
    setSelectedPatient,
    setCgmHistory,
    setIsLoading,
  } = useGlucoTwinStore();

  useEffect(() => {
    fetchPatients()
      .then((data) => {
        setPatients(data);
        if (data.length > 0 && !selectedPatient) {
          handleSelectPatient(data[0]);
        }
      })
      .catch((err) => console.error("Error loading personas:", err));
  }, []);

  const handleSelectPatient = async (patient: PatientProfile) => {
    setSelectedPatient(patient);
    setIsLoading(true);
    try {
      const cgm = await fetchPatientCGM(patient.id, 2);
      setCgmHistory(cgm);
    } catch (err) {
      console.error("Error updating patient data:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-xl font-bold text-text-primary">Interactive What-If Physiology Simulator</h1>
        <p className="text-xs text-text-secondary">
          Simulate postprandial glucose excursions across traditional Indian diets and test the physiological efficacy of activity interventions.
        </p>
      </div>

      <PatientSelector
        patients={patients}
        selectedPatient={selectedPatient}
        onSelect={handleSelectPatient}
      />

      <WhatIfSimulator />
    </div>
  );
}
