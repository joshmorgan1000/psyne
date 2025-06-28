// Jest setup file for Psyne JavaScript bindings tests

// Mock console methods to reduce noise during testing
global.console = {
  ...console,
  log: jest.fn(),
  warn: jest.fn(),
  error: jest.fn(),
};

// Setup test environment
beforeAll(() => {
  // Any global setup needed for tests
});

afterAll(() => {
  // Cleanup after all tests
});

// Increase timeout for integration tests
jest.setTimeout(10000);